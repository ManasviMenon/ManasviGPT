from dotenv import load_dotenv # type: ignore
import os

load_dotenv()

print(os.getenv("GROQ_API_KEY"))
