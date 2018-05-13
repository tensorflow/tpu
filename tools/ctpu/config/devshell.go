// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

package config

import (
	"fmt"
	"log"
	"os"
	"strings"
)

func isDevshell() bool {
	for _, e := range os.Environ() {
		pair := strings.Split(e, "=")
		if pair[0] == "DEVSHELL_CLIENT_PORT" {
			return true
		}
	}
	return false
}

func devshellEnvParseError(e string) error {
	return fmt.Errorf("devshell: unexpected environment variable value: %q", e)
}

func devshellConfig() (*Config, error) {
	cfg := Config{
		Environment: "devshell",
	}
	for _, e := range os.Environ() {
		pair := strings.Split(e, "=")
		switch pair[0] {
		case "DEVSHELL_PROJECT_ID":
			if len(pair) != 2 {
				return nil, devshellEnvParseError(e)
			}
			cfg.Project = pair[1]
		case "DEVSHELL_GCLOUD_CONFIG":
			if len(pair) != 2 {
				return nil, devshellEnvParseError(e)
			}
			cfg.ActiveConfiguration = pair[1]
		default:
			// Nothing
		}
	}

	// TODO(saeta): Investigate falling back to the GCLOUDSDK_CONFIG env var.
	if cfg.Project == "" {
		log.Printf("WARNING: devshell: could not find DEVSHELL_PROJECT_ID")
	}
	return &cfg, nil
}
