from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, TranslationLanguageNotAvailable

def FetchTranscript(id):
    vid = id
    try:
        transcript_api = YouTubeTranscriptApi()
        transcript_list = transcript_api.fetch(video_id=vid, languages=["en"])
        # print(transcript_list)
        transcript = " ".join([chunk.text for chunk in transcript_list])
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound, TranslationLanguageNotAvailable) as E:
        print(f"No Captions available for this video. Error = {E}")
        return ""