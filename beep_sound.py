# from pydub import AudioSegment
# from pydub.playback import _play_with_simpleaudio
# import time

# # Load the sound file (ensure it exists at this path)
# sound = AudioSegment.from_file('//Users/elenikonstantinidou/Downloads/beep_sound.wav')

# # Play the sound
# play_obj = _play_with_simpleaudio(sound)

# # Let it play for 0.5 seconds
# time.sleep(0.5)

# # Stop the sound
# play_obj.stop()


import wave
import simpleaudio as sa

def play_wav_file(file_path):
    try:
        # Open the .wav file
        with wave.open(file_path, 'rb') as wav_file:
            # Read the parameters of the audio
            sample_rate = wav_file.getframerate()
            num_frames = int(sample_rate * 0.5)  # Calculate frames for 0.5 seconds
            
            # Read the first 0.5 seconds of data
            data = wav_file.readframes(num_frames)

        # Play the audio data
        play_obj = sa.play_buffer(data, 1, 2, sample_rate)
        play_obj.wait_done()  # Wait until playback finishes

    except Exception as e:
        print(f"Error: {e}")

play_wav_file('/Users/elenikonstantinidou/Downloads/beep_sound.wav')
