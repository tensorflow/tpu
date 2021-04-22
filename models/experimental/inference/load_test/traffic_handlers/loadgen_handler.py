# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""LoadGen based traffic handler."""

import queue
import tempfile
from typing import Iterable
from absl import logging
import numpy as np

from load_test.traffic_handlers import traffic_handler
import loadgen as lg


class LoadGenHandler(traffic_handler.TrafficHandler):
  """LoadGen based traffic handler."""

  def __init__(self,
               scenario: str,
               total_sample_count: int = 10,
               performance_sample_count: int = 10,
               **kwargs):
    super().__init__(**kwargs)
    self._scenario = scenario
    self._sample_map = {}
    self.total_sample_count = total_sample_count
    self.performance_sample_count = performance_sample_count

  def start(self):
    """Starts the load test."""
    settings = lg.TestSettings()
    if self._scenario == "single_stream":
      settings.scenario = lg.TestScenario.SingleStream
    elif self._scenario == "multi_stream":
      settings.scenario = lg.TestScenario.MultiStream
    elif self._scenario == "server":
      settings.scenario = lg.TestScenario.Server
    else:
      raise ValueError("Unsupported scenario.")
    settings.scenario = lg.TestScenario.MultiStream
    settings.mode = lg.TestMode.PerformanceOnly
    settings.single_stream_expected_latency_ns = 1000000
    settings.min_query_count = 10
    settings.max_query_count = 15
    settings.min_duration_ms = 10000

    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = tempfile.mkdtemp()
    log_settings.log_output.copy_detail_to_stdout = True
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.enable_trace = False

    logging.info("Constructing SUT.")
    sut = lg.ConstructSUT(
        self.issue_queries,
        self.flush_queries,
        self.process_metrics)
    logging.info("Constructing QSL.")
    qsl = lg.ConstructQSL(
        self.total_sample_count,
        self.performance_sample_count,
        self.load_samples,
        self.unload_samples)
    logging.info("Starting test.")
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

  def load_samples(self, sample_indices: Iterable[lg.QuerySample]):
    """Loads samples into RAM."""
    for sample_index in sample_indices:
      if sample_index not in self._sample_map:
        self._sample_map[sample_index] = self.target.prepare(
            self.data_loader.get_sample(sample_index))

  def unload_samples(self, sample_indices: Iterable[lg.QuerySample]):
    for sample_index in sample_indices:
      if sample_index in self._sample_map:
        del self._sample_map[sample_index]

  def issue_queries(self, queries: Iterable[lg.QuerySample]):
    """Sends queries to the system under test.

    This load test creates individual workers for each request, and marks
    completion once all workers complete.

    Args:
      queries: A list of `QuerySample`s.

    """
    worker_queue = queue.Queue()
    responses = []
    def on_completion(query: lg.QuerySample):
      responses.append(lg.QuerySampleResponse(query.id, 0, 0))
      worker_queue.get()
      worker_queue.task_done()

    for query in queries:
      query_id = query.index
      worker_queue.put(query)
      self.target.send(
          query=self._sample_map[query_id],
          completion_callback=on_completion,
          query_handle=query)

    # Wait until workers are complete
    worker_queue.join()
    lg.QuerySamplesComplete(responses)

  def flush_queries(self):
    """Flushes queries, if applicable."""
    self.target.flush()

  def process_metrics(self, latencies_ns: Iterable[float]):
    """Processes the latencies."""
    logging.info("latencies: [p50: %.5f p90:%.5f p99:%.5f]",
                 np.percentile(latencies_ns, 50),
                 np.percentile(latencies_ns, 90),
                 np.percentile(latencies_ns, 99))
