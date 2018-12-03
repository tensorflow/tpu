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

package commands

import (
	"bytes"
	"context"
	"fmt"
	"reflect"
	"testing"

	"cloud.google.com/go/storage"
	"flag"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/cloudresourcemanager/v1beta1"
)

func TestServiceAccounts(t *testing.T) {
	p := &cloudresourcemanager.Project{
		ProjectNumber: 1234567890,
	}
	want := "serviceAccount:service-1234567890@cloud-tpu.iam.gserviceaccount.com"
	got := makeTPUServiceName(p)
	if got != want {
		t.Errorf("makeTPUServiceName(%v) = %q, want: %q", p, got, want)
	}
}

func TestFilterBindings(t *testing.T) {
	sa := "testSa"

	p := &cloudresourcemanager.Policy{
		Bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Members: []string{"foo", "bar", sa},
				Role:    "testrole1",
			},
			&cloudresourcemanager.Binding{
				Members: []string{"baz"},
				Role:    "testrole2",
			},
		},
	}
	bindings := filterBindings(sa, p)

	if len(bindings) != 1 {
		t.Errorf("len(filterBindings(%q, %v)) = %d, want: 1", sa, p, len(bindings))
	}

	if bindings[0].Role != "testrole1" {
		t.Errorf("bindings[0].Role = %q, want: %q", bindings[0].Role, "testrole1")
	}
}

type testAuthResourceManagementCP struct {
	t              *testing.T
	projectNumber  int64
	startPolicy    *cloudresourcemanager.Policy
	capturedPolicy *cloudresourcemanager.Policy
	acls           []storage.ACLRule
	entity         *storage.ACLEntity
	role           *storage.ACLRole
}

func (t *testAuthResourceManagementCP) SA() string {
	return makeTPUServiceName(&cloudresourcemanager.Project{ProjectNumber: t.projectNumber})
}

func (t *testAuthResourceManagementCP) GetProject() (*cloudresourcemanager.Project, error) {
	return &cloudresourcemanager.Project{ProjectNumber: t.projectNumber}, nil
}

func (t *testAuthResourceManagementCP) GetProjectPolicy() (*cloudresourcemanager.Policy, error) {
	return t.startPolicy, nil
}

func (t *testAuthResourceManagementCP) SetProjectPolicy(policy *cloudresourcemanager.Policy) error {
	t.capturedPolicy = policy
	return nil
}

func (t *testAuthResourceManagementCP) GetBucketACL(ctx context.Context, bucket string) ([]storage.ACLRule, error) {
	return t.acls, nil
}

func (t *testAuthResourceManagementCP) SetBucketACL(ctx context.Context, bucket string, entity storage.ACLEntity, role storage.ACLRole) error {
	if t.entity != nil {
		t.t.Errorf("Entity (%v) was not nil when trying to set bucket ACL with %v, %v", t.entity, entity, role)
	}
	if t.role != nil {
		t.t.Errorf("Role (%v) was not nil when trying to set bucket ACL with %v, %v", t.role, entity, role)
	}
	t.entity = &entity
	t.role = &role
	return nil
}

func printBindings(bindings []*cloudresourcemanager.Binding) string {
	var buf bytes.Buffer
	buf.WriteString("[")
	for i, b := range bindings {
		if i != 0 {
			buf.WriteString(", ")
		}
		if b != nil {
			buf.WriteString(fmt.Sprintf("%+v", *b))
		} else {
			buf.WriteString("<nil>")
		}
	}
	buf.WriteString("]")
	return buf.String()
}

func TestAddBigtable(t *testing.T) {
	ctx := context.Background()
	var projectNumber int64 = 123456789
	testCp := testAuthResourceManagementCP{projectNumber: projectNumber}

	testcases := []struct {
		name     string
		input    []*cloudresourcemanager.Binding
		want     []*cloudresourcemanager.Binding
		readonly bool
	}{{
		name: "already_present",
		input: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    bigtableRole,
				Members: []string{"foo", "bar", testCp.SA()},
			},
		},
	}, {
		name: "already_present",
		input: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    bigtableReadonlyRole,
				Members: []string{"foo", "bar", testCp.SA()},
			},
		},
		readonly: true,
	}, {
		name: "already_present_read_write",
		input: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    bigtableRole,
				Members: []string{"foo", "bar", testCp.SA()},
			},
		},
		readonly: true,
	}, {
		name: "must_add_readonly_present",
		input: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    bigtableReadonlyRole,
				Members: []string{"foo", "bar", testCp.SA()},
			},
		},
		want: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    bigtableReadonlyRole,
				Members: []string{"foo", "bar", testCp.SA()},
			},
			&cloudresourcemanager.Binding{
				Role:    bigtableRole,
				Members: []string{testCp.SA()},
			},
		},
	}, {
		name:  "nothing",
		input: []*cloudresourcemanager.Binding{},
		want: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    bigtableRole,
				Members: []string{testCp.SA()},
			},
		},
	}, {
		name:  "nothing",
		input: []*cloudresourcemanager.Binding{},
		want: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    bigtableReadonlyRole,
				Members: []string{testCp.SA()},
			},
		},
		readonly: true,
	}}

	for i, tt := range testcases {
		t.Run(fmt.Sprintf("%d-%s-%v", i, tt.name, tt.readonly), func(t *testing.T) {
			cfg := &config.Config{}
			flags := flag.NewFlagSet("foo", flag.ContinueOnError)
			policy := &cloudresourcemanager.Policy{Bindings: tt.input}
			cp := &testAuthResourceManagementCP{t: t, projectNumber: projectNumber, startPolicy: policy}
			cmd := &authAddBigtable{cfg: cfg, cp: cp, readonly: tt.readonly, skipConf: true}
			got := cmd.Execute(ctx, flags)
			if got != subcommands.ExitSuccess {
				t.Fatalf("cmd.Execute(...) = %v, want: %v", got, subcommands.ExitSuccess)
			}
			if tt.want != nil {
				if !reflect.DeepEqual(tt.want, cp.capturedPolicy.Bindings) {
					t.Errorf("cp.capturedPolicy.Bindings = %v, want: %v", printBindings(cp.capturedPolicy.Bindings), printBindings(tt.want))
				}
			} else {
				if cp.capturedPolicy != nil {
					t.Errorf("cp.capturedPolicy = %v, want: nil", cp.capturedPolicy)
				}
			}
		})
	}
}

func TestAddGcs(t *testing.T) {
	ctx := context.Background()
	var projectNumber int64 = 123456789
	testCp := testAuthResourceManagementCP{projectNumber: projectNumber}
	testEntity := storage.ACLEntity("user-service-123456789@cloud-tpu.iam.gserviceaccount.com")

	testcases := []struct {
		name         string
		bucket       string
		acls         []storage.ACLRule
		bindings     []*cloudresourcemanager.Binding
		readonly     bool
		wantBindings []*cloudresourcemanager.Binding
		wantEntity   storage.ACLEntity
		wantRole     storage.ACLRole
	}{{
		name:   "already_present_global",
		bucket: "",
		acls:   nil,
		bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    storageRole,
				Members: []string{"foo", "bar", testCp.SA()},
			},
		},
	}, {
		name:   "already_present_bucket",
		bucket: "foo",
		acls:   []storage.ACLRule{{Role: storage.RoleOwner, Entity: testEntity}},
		bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    storageRole,
				Members: []string{"foo", "bar"},
			},
		},
	}, {
		name:   "already_present_bucket_role",
		bucket: "foo",
		acls:   []storage.ACLRule{},
		bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    storageRole,
				Members: []string{"foo", "bar", testCp.SA()},
			},
		},
	}, {
		name:       "add_bucket_acls",
		bucket:     "foo",
		acls:       []storage.ACLRule{},
		bindings:   []*cloudresourcemanager.Binding{},
		wantRole:   storage.RoleOwner,
		wantEntity: testEntity,
	}, {
		name:       "add_bucket_readonly",
		bucket:     "foo",
		acls:       []storage.ACLRule{},
		bindings:   []*cloudresourcemanager.Binding{},
		wantRole:   storage.RoleReader,
		wantEntity: testEntity,
		readonly:   true,
	}, {
		name: "modify_bindings",
		bindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    "arbitrary/other.role",
				Members: []string{"foo", "bar"},
			},
			&cloudresourcemanager.Binding{
				Role:    storageRole,
				Members: []string{"baz"},
			},
		},
		wantBindings: []*cloudresourcemanager.Binding{
			&cloudresourcemanager.Binding{
				Role:    "arbitrary/other.role",
				Members: []string{"foo", "bar"},
			},
			&cloudresourcemanager.Binding{
				Role:    storageRole,
				Members: []string{"baz", testCp.SA()},
			},
		},
	}}

	for i, tt := range testcases {
		t.Run(fmt.Sprintf("%d-%s", i, tt.name), func(t *testing.T) {
			cfg := &config.Config{}
			flags := flag.NewFlagSet("foo", flag.ContinueOnError)
			if tt.bucket != "" {
				flags.Parse([]string{tt.bucket})
			}
			policy := &cloudresourcemanager.Policy{Bindings: tt.bindings}
			cp := &testAuthResourceManagementCP{
				t:             t,
				projectNumber: projectNumber,
				startPolicy:   policy,
				acls:          tt.acls,
			}
			cmd := &authAddGcs{cfg: cfg, cp: cp, readonly: tt.readonly, skipConf: true}
			got := cmd.Execute(ctx, flags)
			if got != subcommands.ExitSuccess {
				t.Fatalf("cmd.Execute(...) = %v, want: %v", got, subcommands.ExitSuccess)
			}
			if tt.wantBindings != nil {
				if !reflect.DeepEqual(tt.wantBindings, cp.capturedPolicy.Bindings) {
					t.Errorf("cp.capturedPolicy.Bindings = %v, want: %v", printBindings(cp.capturedPolicy.Bindings), printBindings(tt.wantBindings))
				}
			} else {
				if cp.capturedPolicy != nil {
					t.Errorf("cp.capturedPolicy = %v, want: nil", cp.capturedPolicy)
				}
			}
			if tt.wantEntity != storage.ACLEntity("") {
				if cp.entity == nil || *cp.entity != tt.wantEntity {
					t.Errorf("cp.entity = %v, want: %v", cp.entity, tt.wantEntity)
				}
			} else {
				if cp.entity != nil {
					t.Errorf("cp.entity = %v, want: nil", *cp.entity)
				}
			}
			if tt.wantRole != storage.ACLRole("") {
				if cp.role == nil || tt.wantRole != *cp.role {
					t.Errorf("cp.role = %v, want: %v", cp.role, tt.wantRole)
				}
			} else {
				if cp.role != nil {
					t.Errorf("cp.role = %v, want: nil", *cp.role)
				}
			}
		})
	}
}
