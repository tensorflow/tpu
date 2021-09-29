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

	"google.golang.org/api/cloudresourcemanager/v1beta1"
)

const sampleEtag = "ab12e56"
const sampleVersion = 51
const sampleTPUServiceAccount = "compute-123987@compute.gserviceaccounts.com"

func TestBindingPresent(t *testing.T) {
	policy := &cloudresourcemanager.Policy{
		Etag:    sampleEtag,
		Version: sampleVersion,
		Bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role: "roles/owner",
				Members: []string{
					"user:user1@example.com",
				},
			},
			&cloudresourcemanager.Binding{
				Role: "roles/editor",
				Members: []string{
					"user:user2@example.com",
				},
			},
			&cloudresourcemanager.Binding{
				Role: "roles/logging.logWriter",
				Members: []string{
					"serviceAccount:" + sampleTPUServiceAccount,
				},
			},
			&cloudresourcemanager.Binding{
				Role: "roles/storage.objectAdmin",
				Members: []string{
					"serviceAccount:" + sampleTPUServiceAccount,
				},
			},
		},
	}
	cp := &ResourceManagementCP{}
	modifiedPolicy := cp.addAgentToPolicy(sampleTPUServiceAccount, policy)
	if modifiedPolicy != nil {
		t.Errorf("modified policy was not nil, got: %#v", modifiedPolicy)
	}
}

func TestBindingAbsent(t *testing.T) {
	policy := &cloudresourcemanager.Policy{
		Etag:    sampleEtag,
		Version: sampleVersion,
		Bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role: "roles/owner",
				Members: []string{
					"user:user1@example.com",
				},
			},
			&cloudresourcemanager.Binding{
				Role: "roles/editor",
				Members: []string{
					"user:user2@example.com",
				},
			},
			&cloudresourcemanager.Binding{
				Role: "roles/storage.admin",
				Members: []string{
					"domain:example.com",
				},
			},
		},
	}
	cp := &ResourceManagementCP{}
	modifiedPolicy := cp.addAgentToPolicy(sampleTPUServiceAccount, policy)
	verifyState(t, sampleEtag, sampleVersion, modifiedPolicy)

	b := findBindingForRole(modifiedPolicy, "roles/owner")
	if b == nil {
		t.Errorf("No roles/owner found in modified policy: %#v", modifiedPolicy)
	} else {
		if b.Members == nil || len(b.Members) != 1 || b.Members[0] != "user:user1@example.com" {
			t.Errorf("Members for roles/owner incorrect, got: %#v, want: [user:user1@example.com]", b.Members)
		}
	}
	b = findBindingForRole(modifiedPolicy, "roles/editor")
	if b == nil {
		t.Errorf("No roles/editor found in modified policy: %#v", modifiedPolicy)
	} else {
		if b.Members == nil || len(b.Members) != 1 || b.Members[0] != "user:user2@example.com" {
			t.Errorf("Members for roles/editor incorrect, got: %#v, want: [user:user2@example.com]", b.Members)
		}
	}
	b = findBindingForRole(modifiedPolicy, "roles/storage.admin")
	if b == nil {
		t.Errorf("No roles/storage.admin found in modified policy: %#v", modifiedPolicy)
	} else {
		if b.Members == nil || len(b.Members) != 2 || b.Members[0] != "domain:example.com" || b.Members[1] != "serviceAccount:"+sampleTPUServiceAccount {
			t.Errorf("Members for roles/storage.admin incorrect, got: %#v, want: [domain:example.com, serviceAccount:compute-123987@compute.gserviceaccounts.com]", b.Members)
		}
	}
	b = findBindingForRole(modifiedPolicy, "roles/logging.logWriter")
	if b == nil {
		t.Errorf("No roles/logging.logWriter found in modified policy: %#v", modifiedPolicy)
	} else {
		if b.Members == nil || len(b.Members) != 1 || b.Members[0] != "serviceAccount:"+sampleTPUServiceAccount {
			t.Errorf("Members for roles/logging.logWriter incorrect, got: %#v, want: [serviceAccount:compute-123987@compute.gserviceaccount.com]", b.Members)
		}
	}
}

func TestPartialBinding(t *testing.T) {
	policy := &cloudresourcemanager.Policy{
		Etag:    sampleEtag,
		Version: sampleVersion,
		Bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role: "roles/owner",
				Members: []string{
					"user:noreply@google.com",
				},
			},
			&cloudresourcemanager.Binding{
				Role: "roles/editor",
				Members: []string{
					"serviceAccount:" + sampleTPUServiceAccount,
				},
			},
		},
	}
	cp := &ResourceManagementCP{}
	modifiedPolicy := cp.addAgentToPolicy(sampleTPUServiceAccount, policy)
	if modifiedPolicy != nil {
		t.Errorf("modified policy was not nil, got: %#v", modifiedPolicy)
	}
}

func TestBindingPresentServiceRoleOnly(t *testing.T) {
	policy := &cloudresourcemanager.Policy{
		Etag:    sampleEtag,
		Version: sampleVersion,
		Bindings: []*cloudresourcemanager.Binding{{
			Role:    "roles/owner",
			Members: []string{"user:noreply@google.com"},
		}, {
			Role:    "roles/tpu.serviceAgent",
			Members: []string{"serviceAccount:" + sampleTPUServiceAccount},
		}},
	}
	cp := &ResourceManagementCP{}
	modifiedPolicy := cp.addAgentToPolicy(sampleTPUServiceAccount, policy)

	b := findBindingForRole(modifiedPolicy, "roles/storage.admin")
	if b == nil {
		t.Errorf("No roles/storage.admin found in modified policy: %#v", modifiedPolicy)
	} else {
		if b.Members == nil || len(b.Members) != 1 || b.Members[0] != "serviceAccount:"+sampleTPUServiceAccount {
			t.Errorf("Members for roles/storage.admin incorrect, got: %#v, want: [serviceAccount:compute-123987@compute.gserviceaccounts.com]", b.Members)
		}
	}
	b = findBindingForRole(modifiedPolicy, "roles/logging.logWriter")
	if b == nil {
		t.Errorf("No roles/logging.logWriter found in modified policy: %#v", modifiedPolicy)
	} else {
		if b.Members == nil || len(b.Members) != 1 || b.Members[0] != "serviceAccount:"+sampleTPUServiceAccount {
			t.Errorf("Members for roles/logging.logWriter incorrect, got: %#v, want: [serviceAccount:compute-123987@compute.gserviceaccount.com]", b.Members)
		}
	}
}

func findBindingForRole(policy *cloudresourcemanager.Policy, role string) *cloudresourcemanager.Binding {
	for _, binding := range policy.Bindings {
		if binding.Role == role {
			return binding
		}
	}
	return nil
}

func verifyState(t *testing.T, originalEtag string, originalVersion int64, modified *cloudresourcemanager.Policy) {
	if originalEtag != modified.Etag {
		t.Errorf("policy etag was modified, got: %s, want: %s", modified.Etag, originalEtag)
	}
	if modified.Etag == "" {
		t.Errorf("policy does not have an etag, got: %s", modified.Etag)
	}
	if originalVersion != modified.Version {
		t.Errorf("policy version was modified, got: %d, want: %d", modified.Version, originalVersion)
	}
	if modified.Version == 0 {
		t.Errorf("policy version should be non-zero, got: %d", modified.Version)
	}
}
