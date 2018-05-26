
#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the REST API for async
batch processing.

Example usage:
    python transcribe_async.py resources/audio.raw
    python transcribe_async.py gs://cloud-samples-tests/speech/vr.flac
"""

import argparse
import csv
import datetime
import io
import os
import subprocess

from google.cloud import speech_v1p1beta1 as speech
# from google.cloud import speech
from google.cloud import storage

UPLOAD_BUCKET_NAME = 'bjoeris-temp-audio'

def _safe_filename(filename):
        """
        Generates a safe filename that is unlikely to collide with existing objects
        in Google Cloud Storage.
        ``filename.ext`` is transformed into ``filename-YYYY-MM-DD-HHMMSS.ext``
        """
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
        basename, extension = os.path.splitext(os.path.basename(filename))
        return "{0}-{1}.{2}".format(basename, date, extension)

# [START def_transcribe_file]
def transcribe_file(filename, output):
    """Transcribe the given audio file asynchronously."""
    client = storage.Client()

    print("Converting file...")
    filename = transcode_file(filename)

    bucket_name = UPLOAD_BUCKET_NAME
    bucket = client.bucket(bucket_name)
    blob_name = _safe_filename(filename)
    blob = bucket.blob(blob_name)
    uri = "gs://{}/{}".format(bucket_name, blob_name)
    print("Uploading file...", uri)
    with io.open(filename, 'rb') as audio_file:
        blob.upload_from_file(audio_file)

    operation = transcribe_gcs(uri, output)
    def callback(operation_future):
        print("Deleting file...")
        blob.delete()
    operation.add_done_callback(callback)
    return operation
# [END def_transcribe_file]

def transcode_file(filename):
    stripped_name, ext = os.path.splitext(filename)
    output = '{}-transcode.flac'.format(stripped_name)
    subprocess.run(['ffmpeg', '-i', filename, '-ac', '1', '-ar', '48000', '-acodec', 'flac', output])
    print("transcoded: ", output)
    return output


def transcribe_gcs(gcs_uri, output):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    client = speech.SpeechClient()

    audio = speech.types.RecognitionAudio(uri=gcs_uri)

    metadata = speech.types.RecognitionMetadata()
    metadata.interaction_type = speech.enums.RecognitionMetadata.InteractionType.DISCUSSION
    metadata.microphone_distance = speech.enums.RecognitionMetadata.MicrophoneDistance.NEARFIELD
    metadata.recording_device_type = speech.enums.RecognitionMetadata.RecordingDeviceType.PC

    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=48000,
        language_code='en-US',
        metadata=metadata,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
    )

    print('Transcribing... {}'.format(gcs_uri))
    operation = client.long_running_recognize(config, audio)
    operation.add_done_callback(lambda operation_future: save_results(operation_future.result().results, output))
    return operation

def save_results(results, output):
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'confidence', 'transcript']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for result in results:
            alternative = result.alternatives[0]
            if len(alternative.words) > 0:
                timestamp = alternative.words[0].start_time
                timestamp = timestamp.seconds + 1e-9*timestamp.nanos
                timestamp_hrs = int(timestamp // 3600)
                timestamp_mins = int((timestamp - timestamp_hrs*3600) // 60)
                timestamp_secs = int(timestamp - timestamp_mins * 60 - timestamp_hrs * 3600)
                timestamp_str = '{:0>2d}:{:0>2d}:{:0>2d}'.format(timestamp_hrs, timestamp_mins, timestamp_secs)
            else:
                timestamp_str = ''
            csvwriter.writerow({
                'timestamp': timestamp_str,
                'confidence': '{:.2f}'.format(alternative.confidence),
                'transcript': alternative.transcript,
            })
            print(u'{} | {:.2f} | {}'.format(timestamp_str, alternative.confidence, alternative.transcript))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'audio_file', help='File or GCS path for audio file to be transcribed')
    parser.add_argument(
        '--out', help='File to save the results (CSV)')
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.splitext(args.audio_file)[0] + ".csv"
    operation = None
    if args.audio_file.startswith('gs://'):
        operation = transcribe_gcs(args.audio_file, args.out)
    else:
        operation = transcribe_file(args.audio_file, args.out)
    operation.result()
