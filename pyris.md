# Pyris

Pyris (Python + Iris) is a Python vision and STT library powered by vision LLMs and Whisper.


### How it works

Pyris queries the specified media provider for all or part of a piece of media and routes it to the relevant tool (vision LLM or Whisper) with the user's prompt.

Requests from the user can be: vision only, vision and STT, or STT. Vision only can be a video file or single image. Images must be vision only and audio must be STT.

Pyris has a media provider interface which allows varied audio, video, and image sources. A provider implements method calls to fetch media. For example, a video file provider crops a video at the specified timestamps, captures keyframes at the user specified sample rate (fps), and returns the freezeframe images. An image file provider would just return the image, for now we don't need to worry about compression or cropping. Pyris includes the file provider but the idea with having an exposed API is an application using this library can implement their own providers such as a YouTube video provider, remote url provider, etc.

Video cropping, frame segmentation, and audio extraction from video is powered by ffmpeg. Ffmpeg is called using a subprocess.

The API to this library is async because video processing can take a long time. The output is plaintext.

### Configuration

Pyris can be configured with the following.

- API url | required | the API url for the vision LLM
- Whisper url? (TODO: is there a good solution to host this with an OpenAI API?)
- Model name | required | the driver model used, must support tool use
- Vision model name | optional if not included model name must also support vision | the vision LLM model
- STT model name | optional for startup, but required at runtime for STT | the STT model name for Whisper to use
- System prompt | optional, a backup is included with Pyris | the md file with the system prompt to use for Pyris