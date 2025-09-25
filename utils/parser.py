from langchain_core.output_parsers import StrOutputParser

def parser() -> StrOutputParser:
    parser = StrOutputParser()
    return parser