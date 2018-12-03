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
	"context"
	"fmt"
	"log"
	"net/http"

	"cloud.google.com/go/storage"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/cloudresourcemanager/v1beta1"
	"google.golang.org/api/option"
)

const loggingRole = "roles/logging.logWriter"
const storageRole = "roles/storage.admin" // Note storage.objectAdmin does not work in certain cases, and thus we need roles/storage.admin.
const tpuServiceAgent = "roles/tpu.serviceAgent"

// ResourceManagementCP contains an abstract representation of the Cloud Resource Manager, and related ACLs
//
// It is intentionally small so that other packages in the ctpu tool can be effectively tested.
type ResourceManagementCP struct {
	config  *config.Config
	service *cloudresourcemanager.Service
	storage *storage.Client
}

func newResourceManagementCP(ctx context.Context, config *config.Config, client *http.Client, userAgent string) (*ResourceManagementCP, error) {
	service, err := cloudresourcemanager.New(client)
	if err != nil {
		return nil, err
	}
	service.UserAgent = userAgent
	gcsClient, err := storage.NewClient(ctx, option.WithUserAgent(userAgent), option.WithHTTPClient(client))
	if err != nil {
		return nil, err
	}

	return &ResourceManagementCP{
		config:  config,
		service: service,
		storage: gcsClient,
	}, nil
}

// Adds the tpuUserAgent to the policy and return it. Return nil if there is no need to update the policy.
func (r *ResourceManagementCP) addAgentToPolicy(tpuUserAgent string, policy *cloudresourcemanager.Policy) *cloudresourcemanager.Policy {
	tpuMemberStr := fmt.Sprintf("serviceAccount:%s", tpuUserAgent)

	var loggingBinding, storageBinding *cloudresourcemanager.Binding

	for _, binding := range policy.Bindings {
		if binding.Role == loggingRole {
			loggingBinding = binding
		}
		if binding.Role == storageRole {
			storageBinding = binding
		}
		// Skip checking bindings if this is the tpuServiceAgent role.
		if binding.Role != tpuServiceAgent {
			// Check if the tpuMemberStr is already in a binding.
			for _, member := range binding.Members {
				if member == tpuMemberStr {
					// The TPU service account has already been enabled. Make no modifications.
					return nil
				}
			}
		}
	}

	// Add the tpu service account to the policy and return the policy
	if loggingBinding == nil {
		loggingBinding = &cloudresourcemanager.Binding{Role: loggingRole}
		policy.Bindings = append(policy.Bindings, loggingBinding)
	}
	if storageBinding == nil {
		storageBinding = &cloudresourcemanager.Binding{Role: storageRole}
		policy.Bindings = append(policy.Bindings, storageBinding)
	}
	loggingBinding.Members = append(loggingBinding.Members, tpuMemberStr)
	storageBinding.Members = append(storageBinding.Members, tpuMemberStr)

	return policy
}

// AddTPUUserAgent adds the TPU user agent to enable Cloud Storage access and send logging
//
// It is a no-op if the tpuUserAgent has already been granted some access.
func (r *ResourceManagementCP) AddTPUUserAgent(tpuUserAgent string) error {
	policy, err := r.GetProjectPolicy()
	if err != nil {
		return err
	}
	policy = r.addAgentToPolicy(tpuUserAgent, policy)
	if policy != nil {
		log.Printf("Updating the project's IAM policy to add the TPU service account ('%s') to roles: %s and %s.", tpuUserAgent, loggingRole, storageRole)
		if err := r.SetProjectPolicy(policy); err != nil {
			return err
		}
	}
	return nil
}

// GetProjectPolicy retrieves the IAM policy for the project.
func (r *ResourceManagementCP) GetProjectPolicy() (*cloudresourcemanager.Policy, error) {
	req := cloudresourcemanager.GetIamPolicyRequest{}
	return r.service.Projects.GetIamPolicy(r.config.Project, &req).Do()
}

// SetProjectPolicy sets the IAM policy for project.
func (r *ResourceManagementCP) SetProjectPolicy(policy *cloudresourcemanager.Policy) error {
	req := cloudresourcemanager.SetIamPolicyRequest{Policy: policy}
	_, err := r.service.Projects.SetIamPolicy(r.config.Project, &req).Do()
	return err
}

// GetProject retrieves the project metadata.
func (r *ResourceManagementCP) GetProject() (*cloudresourcemanager.Project, error) {
	return r.service.Projects.Get(r.config.Project).Do()
}

// GetBucketACL retrieves the ACL list for a Cloud Storage bucket.
func (r *ResourceManagementCP) GetBucketACL(ctx context.Context, bucket string) ([]storage.ACLRule, error) {
	bh := r.storage.Bucket(bucket)
	acl := bh.ACL()
	return acl.List(ctx)
}

// SetBucketACL adds the entity to the ACL list at the specified role on the provided bucket.
func (r *ResourceManagementCP) SetBucketACL(ctx context.Context, bucket string, entity storage.ACLEntity, role storage.ACLRole) error {
	bh := r.storage.Bucket(bucket)
	acl := bh.ACL()
	return acl.Set(ctx, entity, role)
}

// IsProjectInGoogleOrg determines if the project is part of the Google organization.
//
// Note: this will need to be updated in the presence of folders.
func (r *ResourceManagementCP) IsProjectInGoogleOrg() (bool, error) {
	resp, err := r.GetProject()
	if err != nil {
		return false, err
	}
	return resp.Parent != nil && resp.Parent.Type == "organization" && resp.Parent.Id == "433637338589", nil
}
