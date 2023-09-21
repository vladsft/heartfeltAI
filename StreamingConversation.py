import asyncio
import signal

import vocode
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber

# these can also be set as environment variables
vocode.setenv(
    OPENAI_API_KEY="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",
    DEEPGRAM_API_KEY="a2aefb2db32d678be7de0427dc07486c0024ef0f",
    AZURE_SPEECH_KEY="ad8f47dbe55b41adadcf74b5a6e23168",
    AZURE_SPEECH_REGION="uksouth",

)


async def main():
    microphone_input, speaker_output = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=True,
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input, endpointing_config=PunctuationEndpointingConfig()
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hello!"),
                prompt_preamble="Have a pleasant conversation about life",
            ),
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(speaker_output)
        ),
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(
        signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
    )
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())