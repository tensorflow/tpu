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

	"flag"
)

func TestFlagConfigAccount(t *testing.T) {
	c := &Config{
		account: "foo@example.com",
	}
	c.computeFlockName()

	if "foo" != c.FlockName {
		t.Errorf("c.FlockName = '%s'; want 'foo'", c.FlockName)
	}
	curUser, _ := user.Current()

	testCases := []struct {
		account  string
		expected string
	}{{
		account:  "foo@example.com",
		expected: "foo",
	}, {
		account:  "@evil.com",
		expected: "",
	}, {
		account:  "s@tooshort.com",
		expected: "",
	}, {
		account:  "x",
		expected: curUser.Username,
	}}

	for _, tt := range testCases {
		c.FlockName = ""
		c.account = tt.account
		c.computeFlockName()
		if c.FlockName != tt.expected {
			t.Errorf("c{account: %q}.FlockName = %q, want: %q", tt.account, c.FlockName, tt.expected)
		}
	}
}

func TestCheckFlockName(t *testing.T) {
	testcases := []struct {
		name    string
		wantErr bool
	}{{
		name:    "goodname",
		wantErr: false,
	}, {
		name:    "good-name",
		wantErr: false,
	}, {
		name:    "bad_name",
		wantErr: true,
	}, {
		name:    "also-bad_",
		wantErr: true,
	}, {
		name:    "_veryBad",
		wantErr: true,
	}}

	for _, test := range testcases {
		err := checkFlockName(test.name)
		gotErr := (err != nil)
		if gotErr != test.wantErr {
			want := "<nil>"
			if test.wantErr {
				want = "<an error>"
			}
			t.Errorf("checkFlockName(%q) = %#v, want: %s", test.name, err, want)
		}
	}
}

func TestCleanFlockName(t *testing.T) {
	testcases := []struct {
		name string
		want string
	}{{
		name: "goodname",
		want: "goodname",
	}, {
		name: "with_underscores",
		want: "with-underscores",
	}, {
		name: "multi__underscores",
		want: "multi-underscores",
	}, {
		name: "foo.bar",
		want: "foo-bar",
	}, {
		name: "-dash",
		want: "dash",
	}, {
		name: "awful . name!",
		want: "awful-name",
	}}

	for _, test := range testcases {
		got := cleanFlockName(test.name)
		if got != test.want {
			t.Errorf("cleanFlockName(%q) = %q, want: %q", test.name, got, test.want)
		}
	}
}

func TestFlagOverrides(t *testing.T) {
	configuredZone := "us-central1-f" // non-default zone
	configuredProject := "my-test-project"
	configuredName := "my-test-flock-name"

	f := flag.NewFlagSet("testFlagset", flag.ContinueOnError)
	cfg := &Config{
		Zone:      configuredZone,
		FlockName: configuredName,
		Project:   configuredProject,
	}

	cfg.SetFlags(f)
	if err := f.Parse(nil); err != nil {
		t.Fatalf("f.Parse(nil) = %v, want nil", err)
	}

	if cfg.Zone != configuredZone {
		t.Errorf("cfg.Zone = %q, want: %q", cfg.Zone, configuredZone)
	}

	if cfg.FlockName != configuredName {
		t.Errorf("cfg.FlockName = %q, want: %q", cfg.FlockName, configuredName)
	}

	if cfg.Project != configuredProject {
		t.Errorf("cfg.Project = %q, want: %q", cfg.Project, configuredProject)
	}
}
