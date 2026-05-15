import json
from src.config import Config
from src.audio_processor import AudioProcessor
from src.llm.llm_analyzer import LLMAnalyzer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def main():
    # 1. Initialization
    print("--- Initializing Pipeline ---")
    audio_proc = AudioProcessor(Config.WHISPER_MODEL_SIZE, Config.TEMP_DIR)
    analyzer = LLMAnalyzer()

    video_input = Config.INPUT_VIDEO_PATH
    if not video_input.exists():
        print(f"Error: Input video not found at {video_input}")
        return

    # 2. Extract and Transcribe
    print(f"--- Extracting Audio and Transcribing: {video_input.name} ---")
    audio_path = audio_proc.extract_audio(video_input)
    transcript_segments = audio_proc.transcribe(audio_path)
    
    # Create Documents from transcript segments, embedding timestamps
    documents = [
        Document(
            page_content=f"[{t['start']:.2f}-{t['end']:.2f}] {t['text']}",
            metadata={'start': t['start'], 'end': t['end']}
        ) 
        for t in transcript_segments
    ]

    # 3. Chunking for LLM analysis
    print("--- Chunking Transcript for LLM Analysis ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.TRANSCRIBE_CHUNK_SIZE,
        chunk_overlap=Config.TRANSCRIBE_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split transcript into {len(chunked_documents)} chunks.")

    # 4. Analyze ALL Chunks with local LLM
    print("--- Analyzing ALL Chunks for Funny Moments ---")
    all_found_moments_data = []
    
    for i, chunk_doc in enumerate(chunked_documents):
        print(f"Analyzing chunk {i+1}/{len(chunked_documents)}...")
        try:
            response = analyzer.analyze_transcript(chunk_doc.page_content)
            
            for moment in response.moments:
                duration = moment.clip_end - moment.clip_start
                # Basic validation for clip duration
                if Config.MIN_CLIP_DURATION <= duration <= Config.MAX_CLIP_DURATION:
                    # Format timestamps to HH:MM:SS
                    start_time_str = f"{int(moment.clip_start // 3600):02d}:" \
                                     f"{int((moment.clip_start % 3600) // 60):02d}:" \
                                     f"{int(moment.clip_start % 60):02d}"
                    end_time_str = f"{int(moment.clip_end // 3600):02d}:" \
                                   f"{int((moment.clip_end % 3600) // 60):02d}:" \
                                   f"{int(moment.clip_end % 60):02d}"

                    all_found_moments_data.append({
                        "start_time": start_time_str,
                        "end_time": end_time_str,
                        "reason": moment.reason
                    })
                    print(f"  [FOUND] {start_time_str} - {end_time_str}: {moment.reason[:50]}...")
                else:
                    print(f"  [SKIPPED] Moment ({duration:.1f}s) outside duration constraints.")

        except Exception as e:
            print(f"Error during LLM analysis for chunk {i+1}: {e}")
            continue 

    if not all_found_moments_data:
        print("No suitable funny moments found in any chunk.")
        return

    # 5. Save moments to JSON file
    output_json_filename = f"{video_input.stem}_funny_moments.json"
    output_json_path = Config.OUTPUT_DIR / output_json_filename
    
    print(f"\n--- Saving {len(all_found_moments_data)} funny moments to {output_json_path} ---")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_found_moments_data, f, indent=4, ensure_ascii=False)

    print(f"\n--- Process Complete! Funny moments saved to: {output_json_path} ---")

if __name__ == "__main__":
    main()
