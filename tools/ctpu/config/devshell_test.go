// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
	"reflect"
	"testing"
)

func TestDevshellFilesystemConfig(t *testing.T) {
	cfgDir := testGcloudConfigDir("no_app_creds")
	env := []string{
		"CLOUDSDK_CONFIG=" + cfgDir,
		"EXTRA_ENV_VAR=foo",
	}
	got := devshellFilesystemConfig(env)
	want := &Config{
		Environment:         "devshell",
		Project:             "ctpu9-test-project",
		account:             "saeta@google.com",
		Zone:                "us-central1-c",
		FlockName:           "",
		ActiveConfiguration: "ctpu9",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("devshellFilesystemConfig(%v) = %#v, want: %#v", env, got, want)
	}
}

func TestDevshellFilesystemConfigEmpty(t *testing.T) {
	cfgDir := testGcloudConfigDir("no_config")
	env := []string{
		"CLOUDSDK_CONFIG=" + cfgDir,
		"EXTRA_ENV_VAR=foo",
	}
	got := devshellFilesystemConfig(env)
	want := &Config{
		Environment: "devshell",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("devshellFilesystemConfig(%v) = %#v, want: %#v", env, got, want)
	}
}

func TestDevshellFilesystemConfigBadEnv(t *testing.T) {
	env := []string{
		"CLOUDSDK_CONFIG=",
		"EXTRA_ENV_VAR=foo",
	}
	got := devshellFilesystemConfig(env)
	if got != nil {
		t.Errorf("devshellFilesystemConfig(%v) = %v, want: nil", env, got)
	}
}
