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

	if cfg.ActiveConfiguration != "ctpu9" {
		t.Error("Active configuration: " + cfg.ActiveConfiguration)
	}
	if cfg.account != "saeta@google.com" {
		t.Error("Account: " + cfg.account)
	}
	if cfg.Project != "ctpu9-test-project" {
		t.Error("Project: " + cfg.Project)
	}
	if cfg.Zone != "us-central1-c" {
		t.Error("Zone: " + cfg.Zone)
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
	if cfg.ActiveConfiguration != "" {
		t.Error("Active config was non-empty! " + cfg.ActiveConfiguration)
	}
	if cfg.Project != "" {
		t.Error("project was non-empty! " + cfg.Project)
	}
	if cfg.Zone != "" {
		t.Error("zone was non-empty! " + cfg.Zone)
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

	if cfg.ActiveConfiguration != "ctpu9" {
		t.Error("Active configuration error: " + cfg.ActiveConfiguration)
	}

	if cfg.Project != "" {
		t.Error("Project was non-empty! " + cfg.Project)
	}

	if cfg.Zone != "us-central1-c" {
		t.Error("Zone error: " + cfg.Zone)
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
