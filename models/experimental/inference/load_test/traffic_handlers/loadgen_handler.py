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

import tempfile
import threading
from typing import Iterable
from absl import logging
import numpy as np

from load_test.traffic_handlers import traffic_handler
import mlperf_loadgen as lg


# Default settings based on mlperf.conf:
# https://github.com/mlcommons/inference/blob/master/mlperf.conf

# Single stream
_DEFAULT_SINGLE_STREAM_LATENCY_PERCENTILE = 0.9
_DEFAULT_SINGLE_STREAM_TARGET_LATENCY = 10*int(1e6)
_DEFAULT_SINGLE_STREAM_QUERY_COUNT = 1024

# Multi stream
_DEFAULT_MULTI_STREAM_QPS = 20
_DEFAULT_MULTI_STREAM_QUERY_COUNT = 270336
_DEFAULT_MULTI_STREAM_LATENCY_PERCENTILE = 0.99
_DEFAULT_MULTI_STREAM_TARGET_LATENCY = 50*int(1e6)

# Server
_DEFAULT_SERVER_QPS = 1.0
_DEFAULT_SERVER_QUERY_COUNT = 270336
_DEFAULT_SERVER_LATENCY_PERCENTILE = 0.99
_DEFAULT_SERVER_TARGET_LATENCY = 10*int(1e6)


class LoadGenHandler(traffic_handler.TrafficHandler):
  """LoadGen based traffic handler."""

  def __init__(self,
               scenario: str,
               total_sample_count: int = 10,
               performance_sample_count: int = 10,
               duration_ms: int = None,
               target_latency_percentile: float = None,
               target_latency_ns: int = None,
               query_count: int = None,
               qps: int = None,
               **kwargs):
    super().__init__(**kwargs)
    self._scenario = scenario
    self._sample_map = {}
    self.total_sample_count = total_sample_count
    self.performance_sample_count = performance_sample_count
    self._target_latency_percentile = target_latency_percentile
    self._target_latency_ns = target_latency_ns
    self._query_count = query_count
    self._duration_ms = duration_ms
    self._qps = qps
    self._lock = threading.Lock()
    self._completed_queries = 0
    self._failed_queries = 0

  def get_test_settings(self) -> lg.TestSettings:
    settings = lg.TestSettings()
    settings.min_duration_ms = self._duration_ms or 600000

    if self._scenario == "single_stream":
      settings.scenario = lg.TestScenario.SingleStream
      settings.min_query_count = (
          self._query_count or _DEFAULT_SINGLE_STREAM_QUERY_COUNT)
      settings.single_stream_target_latency_percentile = (
          self._target_latency_percentile or
          _DEFAULT_SINGLE_STREAM_LATENCY_PERCENTILE)

      settings.single_stream_expected_latency_ns = (
          self._target_latency_ns or _DEFAULT_SINGLE_STREAM_TARGET_LATENCY)

    elif self._scenario == "multi_stream":
      settings.scenario = lg.TestScenario.MultiStream
      settings.multi_stream_target_qps = self._qps or _DEFAULT_MULTI_STREAM_QPS

      settings.min_query_count = (
          self._query_count or _DEFAULT_MULTI_STREAM_QUERY_COUNT)
      settings.multi_stream_target_latency_percentile = (
          self._target_latency_percentile
          or _DEFAULT_MULTI_STREAM_LATENCY_PERCENTILE)

      settings.multi_stream_target_latency_ns = (
          self._target_latency_ns or _DEFAULT_MULTI_STREAM_TARGET_LATENCY)

    elif self._scenario == "server":
      settings.scenario = lg.TestScenario.Server
      settings.min_query_count = (
          self._query_count or _DEFAULT_SERVER_QUERY_COUNT)
      settings.server_target_qps = self._qps or _DEFAULT_SERVER_QPS
      settings.server_target_latency_ns = (
          self._target_latency_ns or _DEFAULT_SERVER_TARGET_LATENCY)
      settings.server_target_latency_percentile = (
          self._target_latency_percentile or _DEFAULT_SERVER_LATENCY_PERCENTILE)
    else:
      raise ValueError("Unsupported scenario.")

    settings.mode = lg.TestMode.PerformanceOnly

    return settings

  def start(self):
    """Starts the load test."""
    settings = self.get_test_settings()

    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = tempfile.mkdtemp()
    log_settings.log_output.copy_detail_to_stdout = False
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.enable_trace = False

    logging.info("Constructing SUT.")
    sut = lg.ConstructSUT(
        self.issue_query,
        self.flush_queries,
        self.process_metrics)
    logging.info("Constructing QSL.")
    qsl = lg.ConstructQSL(
        self.total_sample_count,
        self.performance_sample_count or self.total_sample_count,
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

  def issue_query(self, samples: Iterable[lg.QuerySample]):
    """Sends query samples to the system under test.

    Creates a separate thread for each sample in the query.

    Args:
      samples: A list of `QuerySample`s.

    """
    def on_completion(query: lg.QuerySample, success=True):
      response = lg.QuerySampleResponse(query.id, 0, 0)
      lg.QuerySamplesComplete([response])
      with self._lock:
        self._completed_queries += 1
        self._failed_queries += int(not success)

    for sample in samples:
      threading.Thread(
          target=self.target.send,
          kwargs=dict(
              query=self._sample_map[sample.index],
              completion_callback=on_completion,
              query_handle=sample)).start()

  def flush_queries(self):
    """Flushes queries, if applicable."""
    self.target.flush()

  def process_metrics(self, latencies_ns: Iterable[float]):
    """Processes the latencies."""
    logging.info("latencies: [p50: %.5f p90:%.5f p99:%.5f]\n",
                 np.percentile(latencies_ns, 50),
                 np.percentile(latencies_ns, 90),
                 np.percentile(latencies_ns, 99))

    logging.info("Completed Queries: %d, Failed Queries: %d\n",
                 self._completed_queries, self._failed_queries)
