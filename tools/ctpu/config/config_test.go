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
	"os/user"
	"testing"
)

func init() {
	// Set the default flag values for the tests
	RegisterFlags()
}

func TestFlagConfigAccount(t *testing.T) {
	g := envConfig{
		account: "foo@example.com",
		project: "test-project-id",
		zone:    "us-central1-c",
	}
	f, err := newFlagOverrideConfig(&g)
	if err != nil {
		t.Fatal("Error constructing new config.", err)
	}

	if "foo" != f.FlockName() {
		t.Errorf("f.FlockName() = '%s'; want 'foo'", f.FlockName())
	}

	type testCase struct {
		account  string
		expected string
	}
	curUser, _ := user.Current()
	testCases := []testCase{
		{
			account:  "foo@example.com",
			expected: "foo",
		},
		{
			account:  "@evil.com",
			expected: "",
		},
		{
			account:  "s@tooshort.com",
			expected: "",
		},
		{
			account:  "x",
			expected: curUser.Username,
		},
	}

	for _, tt := range testCases {
		g.account = tt.account
		actual, err := newFlagOverrideConfig(&g)
		if tt.expected == "" {
			if err == nil {
				t.Errorf("newFlagOverrideConfig(%q).err == nil, want: non-nil.", tt.account)
			}
		} else {
			if err != nil {
				t.Errorf("newFlagOverrideConfig(%q).err = %v, want: nil", tt.account, err)
			}
			if actual.FlockName() != tt.expected {
				t.Errorf("newFlagOverrideConfig(%q).FlockName() = %q, want: %q", tt.account, actual.FlockName(), tt.expected)
			}
		}
	}

	flockOverride = "short"
	f, err = newFlagOverrideConfig(&g)
	if err != nil {
		t.Fatal("Error constructing override config: ", err)
	}
	if "short" != f.FlockName() {
		t.Errorf("f.FlockName() = '%s', want 'short'", f.FlockName())
	}
}

func TestFlagConfigZone(t *testing.T) {
	g := envConfig{
		account: "foo@example.com",
		project: "test-project-id",
		zone:    "us-central1-c",
	}
	f, err := newFlagOverrideConfig(&g)
	if err != nil {
		t.Fatalf("Error constructing new config: %#v", err)
	}
	if "us-central1-c" != f.Zone() {
		t.Errorf("f.Zone() = '%s', want 'us-central1-c'", f.Zone())
	}

	zoneOverride = "us-central1-f"
	if "us-central1-f" != f.Zone() {
		t.Errorf("f.Zone() = '%s', want 'us-central1-f'", f.Zone())
	}
}

func TestFlagConfigProject(t *testing.T) {
	g := envConfig{
		account: "foo@example.com",
		project: "my-tpu-project",
		zone:    "us-central1-c",
	}
	f, err := newFlagOverrideConfig(&g)
	if err != nil {
		t.Fatalf("Error constructing new config: %#v", err)
	}
	if "my-tpu-project" != f.Project() {
		t.Error(f.Project())
	}
	projectOverride = "my-other-tpu-project"
	if "my-other-tpu-project" != f.Project() {
		t.Errorf("f.Project() = '%s', want: 'my-other-tpu-project'", f.Project())
	}
}

func TestFlagConfigIpRange(t *testing.T) {
	if nil != validateCidrRange() {
		t.Fatal("Could not validate the default network range", validateCidrRange())
	}

	ipRange = "0.0.0.0/0"
	if nil == validateCidrRange() {
		t.Error("Empty network range allowed through!")
	}

	ipRange = "192.168.3.3/32"
	if nil == validateCidrRange() {
		t.Error("Single-IP address range allowed through!")
	}
}
