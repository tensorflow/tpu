# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
from concurrent import futures
import functools
from io import BytesIO  # pylint:disable=g-importing-member
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm


_PROMPTS = [
    "Labrador in the style of Hokusai",
    "Painting of a squirrel skating in New York",
    "HAL-9000 in the style of Van Gogh",
    "Times Square under water, with fish and a dolphin swimming around",
    "Ancient Roman fresco showing a man working on his laptop",
    "Armchair in the shape of an avocado",
    "Clown astronaut in space, with Earth in the background",
    "A cat sitting on a windowsill",
    "A dog playing fetch in a park",
    "A city skyline at night",
    "A field of flowers in bloom",
    "A tropical beach with palm trees",
    "A snowy mountain range",
    "A waterfall cascading into a pool",
    "A forest at sunset",
    "A desert landscape with cacti",
    "A volcano erupting",
    "A lightning storm in the distance",
    "A rainbow over a rainbow",
    "A unicorn grazing in a meadow",
    "A dragon flying through the sky",
    "A mermaid swimming in the ocean",
    "A robot walking down the street",
    "A UFO landing in a field",
    "A portal to another dimension",
    "A time traveler from the future",
    "A talking cat",
    "A bowl of fruit on a table",
    "A group of friends laughing",
    "A family sitting down for dinner",
    "A couple kissing in the rain",
    "A child playing with a toy",
    "A musician playing an instrument",
    "A painter painting a picture",
    "A writer writing a book",
    "A scientist conducting an experiment",
    "A construction worker building a house",
    "A doctor operating on a patient",
    "A teacher teaching a class",
    "A police officer arresting a suspect",
    "A firefighter putting out a fire",
    "A soldier fighting in a war",
    "A farmer working in a field",
    "A pilot flying a plane",
    "An astronaut in space",
    "A unicorn eating a rainbow"
]


def send_request_and_receive_image(prompt: str, url: str) -> BytesIO:
  """Sends a single prompt request and returns the Image."""
  try:
    inputs = "%20".join(prompt.split(" "))
    resp = requests.get(f"{url}?prompt={inputs}")
    resp.raise_for_status()
    return BytesIO(resp.content)
  except requests.RequestException as e:
    print(f"An error occurred while sending the request: {e}")


def image_grid(imgs, rows, cols):
  w, h = imgs[0].size
  grid = Image.new("RGB", size=(cols * w, rows * h))
  for i, img in enumerate(imgs):
    grid.paste(img, box=(i % cols * w, i // cols * h))
  return grid


def send_requests(num_requests: int, batch_size: int, save_pictures: bool,
                  url: str = "http://localhost:8000/imagine"):
  """Sends a list of requests and processes the responses."""
  print("num_requests: ", num_requests)
  print("batch_size: ", batch_size)
  print("url: ", url)
  print("save_pictures: ", save_pictures)

  prompts = _PROMPTS
  if num_requests > len(_PROMPTS):
    # Repeat until larger than num_requests
    prompts = _PROMPTS * int(np.ceil(num_requests / len(_PROMPTS)))

  prompts = np.random.choice(
      prompts, num_requests, replace=False)

  with futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
    raw_images = list(
        tqdm(
            executor.map(
                functools.partial(send_request_and_receive_image, url=url),
                prompts,
            ),
            total=len(prompts),
        )
    )

  if save_pictures:
    print("Saving pictures to diffusion_results.png")
    images = [Image.open(raw_image) for raw_image in raw_images]
    grid = image_grid(images, 2, num_requests // 2)
    grid.save("./diffusion_results.png")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sends requests to Diffusion.")
  parser.add_argument(
      "--num_requests", help="Number of requests to send.",
      default=8)
  parser.add_argument(
      "--batch_size", help="The number of requests to send at a time.",
      default=8)
  parser.add_argument(
      "--save_pictures", default=False, action="store_true",
      help="Whether to save the generated pictures to disk.")
  parser.add_argument(
      "--ip", help="The IP address to send the requests to.")

  args = parser.parse_args()

  address = f"http://{args.ip}:8000/imagine"
  send_requests(
      num_requests=int(args.num_requests), batch_size=int(args.batch_size),
      save_pictures=bool(args.save_pictures), url=address)
