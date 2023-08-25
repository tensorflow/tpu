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
"""Ray Serve Stable Diffusion example."""
from io import BytesIO  # pylint:disable=g-importing-member
from typing import List
from fastapi import FastAPI  # pylint:disable=g-importing-member
from fastapi.responses import Response  # pylint:disable=g-importing-member
from ray import serve


app = FastAPI()
_MAX_BATCH_SIZE = 64


@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
  """`APIIngress`, e.g. the request router.

  Attributes:
    handle: The handle that we use to access the Diffusion
      model server that runs on TPU hardware.

  """

  def __init__(self, diffusion_model_handle) -> None:
    self.handle = diffusion_model_handle

  @app.get(
      "/imagine",
      responses={200: {"content": {"image/png": {}}}},
      response_class=Response,
  )
  async def generate(self, prompt: str):
    """Requests the generation of an individual prompt.

    Args:
      prompt: An individual prompt.

    Returns:
      A Response.

    """
    result_handle = await self.handle.generate.remote(prompt)
    return await result_handle


@serve.deployment(
    ray_actor_options={
        "resources": {"TPU": 1},
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": _MAX_BATCH_SIZE,
    })
class StableDiffusion:
  """FLAX Stable Diffusion Ray Serve deployment running on TPUs.

  Attributes:
    run_with_profiler: Whether or not to run with the profiler. Note that
      this saves the profile to the separate TPU VM.

  """

  def __init__(
      self, run_with_profiler: bool = False, warmup: bool = False,
      warmup_batch_size: int = _MAX_BATCH_SIZE):
    from diffusers import FlaxStableDiffusionPipeline  # pylint:disable=g-import-not-at-top,g-importing-member
    from flax.jax_utils import replicate  # pylint:disable=g-import-not-at-top,g-importing-member
    import jax  # pylint:disable=g-import-not-at-top,unused-import
    import jax.numpy as jnp  # pylint:disable=g-import-not-at-top
    from jax import pmap  # pylint:disable=g-import-not-at-top,g-importing-member

    model_id = "CompVis/stable-diffusion-v1-4"

    self._pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
        model_id,
        revision="bf16",
        dtype=jnp.bfloat16)

    self._p_params = replicate(params)
    self._p_generate = pmap(self._pipeline._generate)
    self._run_with_profiler = run_with_profiler
    self._profiler_dir = "/tmp/tensorboard"

    if warmup:
      print("Sending warmup requests.")
      warmup_prompts = ["A warmup request"] * warmup_batch_size
      self.generate_tpu(warmup_prompts)

  def generate_tpu(self, prompts: List[str]):
    """Generates a batch of images from Diffusion from a list of prompts.

    Args:
      prompts: a list of strings. Should be a factor of 4.

    Returns:
      A list of PIL Images.
    """
    from flax.training.common_utils import shard  # pylint:disable=g-import-not-at-top,g-importing-member
    import jax   # pylint:disable=g-import-not-at-top
    import time  # pylint:disable=g-import-not-at-top
    import numpy as np  # pylint:disable=g-import-not-at-top

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, jax.device_count())

    assert prompts, "prompt parameter cannot be empty"
    print("Prompts: ", prompts)
    prompt_ids = self._pipeline.prepare_inputs(prompts)
    prompt_ids = shard(prompt_ids)
    print("Sharded prompt ids has shape:", prompt_ids.shape)
    if self._run_with_profiler:
      jax.profiler.start_trace(self._profiler_dir)

    time_start = time.time()
    images = self._p_generate(prompt_ids, self._p_params, rng)
    images = images.block_until_ready()
    elapsed = time.time() - time_start
    if self._run_with_profiler:
      jax.profiler.stop_trace()

    print("Inference time (in seconds): ", elapsed)
    print("Shape of the predictions: ", images.shape)
    images = images.reshape(
        (images.shape[0] * images.shape[1],) + images.shape[-3:])
    print("Shape of images afterwards: ", images.shape)
    return self._pipeline.numpy_to_pil(np.array(images))

  @serve.batch(batch_wait_timeout_s=10, max_batch_size=_MAX_BATCH_SIZE)
  async def batched_generate_handler(self, prompts: List[str]):
    """Sends a batch of prompts to the TPU model server.

    This takes advantage of @serve.batch, Ray Serve's built-in batching
    mechanism.

    Args:
      prompts: A list of input prompts

    Returns:
      A list of responses which contents are raw PNG.
    """
    print("Number of input prompts: ", len(prompts))
    num_to_pad = _MAX_BATCH_SIZE - len(prompts)
    prompts += ["Scratch request"] * num_to_pad

    images = self.generate_tpu(prompts)
    results = []
    for image in images[: _MAX_BATCH_SIZE - num_to_pad]:
      file_stream = BytesIO()
      image.save(file_stream, "PNG")
      results.append(
          Response(content=file_stream.getvalue(), media_type="image/png")
      )
    return results

  async def generate(self, prompt):
    return await self.batched_generate_handler(prompt)


diffusion_bound = StableDiffusion.bind()
deployment = APIIngress.bind(diffusion_bound)
