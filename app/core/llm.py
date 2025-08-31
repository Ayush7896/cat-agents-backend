from langchain_openai import ChatOpenAI
from app.core.config import settings

# single shared model exactly like your snippet
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    top_p=0.1,
    max_tokens=800,
    api_key=settings.OPENAI_API_KEY
)
