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

package ctrl

import (
	"testing"
	"time"

	"golang.org/x/oauth2"
)

func TestParseResponse(t *testing.T) {
	addTime, err := time.ParseDuration("1234.2s")
	if err != nil {
		t.Fatal(err)
	}
	testcases := []struct {
		input string
		want  oauth2.Token
	}{{
		input: "[\"saeta@example.com\",\"ctpu-test-project\",\"abc123\",1234.2]",
		want: oauth2.Token{
			AccessToken: "abc123",
			Expiry:      time.Now().Add(addTime),
		},
	}}

	for _, testcase := range testcases {
		src := devshellTokenSource{}
		got, err := src.parseResponse(testcase.input)
		if err != nil {
			t.Fatalf("src.parseResponse(%q) = %v, want nil", testcase.input, err)
		}
		if got == nil {
			t.Fatalf("src.parseResponse(%q) = nil, want non-nil", testcase.input)
		}
		if got.AccessToken != testcase.want.AccessToken {
			t.Errorf("src.parseResponse(%q).AccessToken = %q, want %q", testcase.input, got.AccessToken, testcase.want.AccessToken)
		}
		// Add some slop to avoid test flakiness.
		if got.Expiry.Sub(testcase.want.Expiry) > 200*time.Millisecond {
			t.Errorf("src.parseResponse(%q).Expiry = %v, want: %v", testcase.input, got.Expiry.Truncate(100*time.Millisecond), testcase.want.Expiry.Truncate(100*time.Millisecond))
		}
	}
}
