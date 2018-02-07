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
	"path"
	"strings"
	"testing"
)

func testGcloudConfigDir(testName string) string {
	return path.Join(".", "testdata", "gcloud", testName)
}

func TestGcloudClean(t *testing.T) {
	cfgDir := testGcloudConfigDir("clean")
	cfg, err := buildGcloudEnvConfig(cfgDir)
	if err != nil {
		t.Fatal(err.Error())
	}

	if cfg.activeConfiguration != "ctpu9" {
		t.Error("Active configuration: " + cfg.activeConfiguration)
	}
	if cfg.account != "saeta@google.com" {
		t.Error("Account: " + cfg.account)
	}
	if cfg.project != "ctpu9-test-project" {
		t.Error("Project: " + cfg.project)
	}
	if cfg.zone != "us-central1-c" {
		t.Error("Zone: " + cfg.zone)
	}
}

func TestGcloudCorruptedMissingConfig(t *testing.T) {
	cfgDir := testGcloudConfigDir("corrupted")
	_, err := buildGcloudEnvConfig(cfgDir)

	if err == nil {
		t.Fatal("Corrupted did not encounter an error.")
	}
	if !strings.Contains(err.Error(), "active config file not found") {
		t.Error(err.Error())
	}
}

func TestGcloudCorruptedNoConfigurationsDirectory(t *testing.T) {
	cfgDir := testGcloudConfigDir("corrupted2")
	cfg, err := buildGcloudEnvConfig(cfgDir)

	if err != nil {
		t.Fatal(err)
	}
	if cfg.account != "" {
		t.Error("Account was non-empty! " + cfg.account)
	}
	if cfg.activeConfiguration != "" {
		t.Error("Active config was non-empty! " + cfg.activeConfiguration)
	}
	if cfg.project != "" {
		t.Error("project was non-empty! " + cfg.project)
	}
	if cfg.zone != "" {
		t.Error("zone was non-empty! " + cfg.zone)
	}
}

func TestGcloudIncomplete(t *testing.T) {
	cfgDir := testGcloudConfigDir("incomplete")
	cfg, err := buildGcloudEnvConfig(cfgDir)

	if err != nil {
		t.Fatal(err)
	}

	if cfg.account != "saeta@google.com" {
		t.Error("Account error: " + cfg.account)
	}

	if cfg.activeConfiguration != "ctpu9" {
		t.Error("Active configuration error: " + cfg.activeConfiguration)
	}

	if cfg.project != "" {
		t.Error("Project was non-empty! " + cfg.project)
	}

	if cfg.zone != "us-central1-c" {
		t.Error("Zone error: " + cfg.zone)
	}
}

func TestGcloudNoConfig(t *testing.T) {
	cfgDir := testGcloudConfigDir("no_config")
	_, err := buildGcloudEnvConfig(cfgDir)

	if err == nil {
		t.Fatal(err)
	}
	if !strings.Contains(err.Error(), "no application default credentials found") {
		t.Error(err.Error())
	}
}

func TestGcloudNoDir(t *testing.T) {
	cfgDir := testGcloudConfigDir("not_there")
	_, err := buildGcloudEnvConfig(cfgDir)
	if err == nil {
		t.Fatal(err)
	}
}
