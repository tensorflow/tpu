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
	"os"
	"os/user"
	"path"

	"cloud.google.com/go/compute/metadata"
)

func gceConfig() (*Config, error) {
	cfg := &Config{}

	// Load config from the filesystem if present.
	if user, err := user.Current(); err == nil {
		configDir := path.Join(user.HomeDir, ".config", "gcloud")
		if _, err := os.Stat(configDir); err == nil {
			fsCfg, err := buildGcloudEnvConfig(configDir, false)
			if err == nil {
				cfg = fsCfg
			}
		}
	}

	cfg.Environment = "gce"

	if cfg.Project == "" {
		p, err := metadata.ProjectID()
		if err != nil {
			return nil, err
		}
		cfg.Project = p
	}

	if cfg.Zone == "" {
		z, err := metadata.Zone()
		if err != nil {
			return nil, err
		}
		cfg.Zone = z
	}

	if cfg.FlockName == "" {
		fn, err := metadata.InstanceName()
		if err != nil {
			return nil, err
		}
		cfg.FlockName = fn
	}

	return cfg, nil
}
