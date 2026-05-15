from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.models import FunnyMomentsList
from src.llm.llm_factory import LLMFactory

class LLMAnalyzer:
    def __init__(self):
        # Use the factory to get the LLM instance (LM Studio or NVIDIA)
        self.llm = LLMFactory.get_llm()
        self.parser = PydanticOutputParser(pydantic_object=FunnyMomentsList)

        # Initialize the prompt template and chain once
        self.prompt = ChatPromptTemplate.from_template(
            "คุณคือผู้เชี่ยวชาญด้านการตัดต่อวิดีโอภาษาไทย อ่านส่วนของสคริปต์วิดีโอต่อไปนี้\n"
            "ระบุช่วงเวลาที่ตลกขบขัน มุกตลก หรือบทสนทนาที่น่าสนใจทั้งหมดในข้อความนี้\n"
            "สำหรับแต่ละช่วงเวลา ให้ระบุเวลาเริ่มต้นและสิ้นสุด พร้อมเหตุผลสั้นๆ\n"
            "พิจารณาคำสแลงและบริบทของภาษาไทย คลิปแต่ละคลิปควรมีความยาวประมาณ 15-60 วินาที\n\n"
            "format_instructions: {format_instructions}\n\n"
            "transcript:\n{transcript}"
        )
        
        self.chain = self.prompt | self.llm | self.parser

    def analyze_transcript(self, transcript_text: str) -> FunnyMomentsList:
        result = self.chain.invoke({
            "transcript": transcript_text,
            "format_instructions": self.parser.get_format_instructions()
        })
        return result
