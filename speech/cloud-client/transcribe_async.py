
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

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage

def _safe_filename(filename):
        """
        Generates a safe filename that is unlikely to collide with existing objects
        in Google Cloud Storage.
        ``filename.ext`` is transformed into ``filename-YYYY-MM-DD-HHMMSS.ext``
        """
        date = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
        basename, extension = filename.rsplit('.', 1)
        return "{0}-{1}.{2}".format(basename, date, extension)

# [START def_transcribe_file]
def transcribe_file(filename, output):
    """Transcribe the given audio file asynchronously."""
    client = storage.Client()

    bucket_name = 'bjoeris-temp-audio'
    bucket = client.bucket(bucket_name)
    blob_name = _safe_filename(filename)
    blob = bucket.blob(blob_name)
    print("Uploading file...")
    with io.open(filename, 'rb') as audio_file:
        blob.upload_from_file(audio_file)
    uri = "gs://{}/{}".format(bucket_name, blob_name)

    transcribe_gcs(uri, output)
    print("Deleting file...")
    blob.delete()
# [END def_transcribe_file]


# [START def_transcribe_gcs]
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
        sample_rate_hertz=16000,
        language_code='en-US',
        metadata=metadata,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True)

    operation = client.long_running_recognize(config, audio)

    print('Transcribing...')
    response = operation.result(timeout=90)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    timestamp = 0.0
    with open(output, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'confidence', 'transcript']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for result in response.results:
            alternative = result.alternatives[0]
            if len(alternative.words) > 0:
                timestamp = alternative.words[0].start_time
                timestamp = timestamp.seconds + 1e-9*timestamp.nanos
                timestamp_mins = int(timestamp // 60)
                timestamp_secs = timestamp - timestamp_mins * 60
                csvwriter.writerow({
                    'timestamp': '{}:{}'.format(timestamp_mins, timestamp_secs),
                    'confidence': alternative.confidence,
                    'transcript': alternative.transcript,
                })
                print(u'{}:{} | {} | {}'.format(timestamp_mins, timestamp_secs , alternative.confidence, alternative.transcript))
# [END def_transcribe]


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
    if args.audio_file.startswith('gs://'):
        transcribe_gcs(args.audio_file, args.out)
    else:
        transcribe_file(args.audio_file, args.out)
