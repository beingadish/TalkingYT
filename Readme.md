# YouTube Video Chat

An AI-powered chat system that allows users to interact with YouTube video content through natural language queries. Built using LangChain and Google's Gemini model.

## Features

- Extract transcripts from YouTube videos
- Process and chunk video transcripts
- Generate embeddings using Google's Gemini model
- Create semantic search capabilities using FAISS vector store
- Interactive Q&A with video content
- Multi-query retrieval for better context understanding

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini model)
- Virtual Environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TalkingYoutube
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Project Structure

```
TalkingYoutube/
├── assistant/
│   └── prompt.py           # AI assistant prompt templates
├── indexing/
│   ├── document_load.py    # YouTube transcript fetching
│   ├── embedder.py         # Document embedding generation
│   └── splitter.py         # Text splitting utilities
├── utils/
│   ├── formatter.py        # Context formatting
│   └── parser.py          # Output parsing
├── .env                    # Environment variables
├── main.py                # Main application
└── requirements.txt       # Project dependencies
```

## Usage

1. Run the main script with a YouTube video ID:
```bash
python main.py
```

2. The system will:
   - Fetch the video transcript
   - Split it into manageable chunks
   - Generate embeddings
   - Create a vector store
   - Allow you to ask questions about the video content

## Example

```python
question = "What is vectorization?"
answer = ask.invoke(question)
print(answer)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

[Add your chosen license]

## Acknowledgments

- LangChain for the framework
- Google's Gemini model for embeddings and chat
- FAISS for vector storage
- YouTube Transcript API for caption extraction