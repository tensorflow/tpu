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
	"fmt"
	"log"
	"net/http"

	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/cloudresourcemanager/v1beta1"
)

const loggingRole = "roles/logging.logWriter"
const storageRole = "roles/storage.admin" // Note storage.objectAdmin does not work in certain cases, and thus we need roles/storage.admin.

// ResourceManagementCP contains an abstract representation of the Cloud Resource Manager, and related ACL's
//
// It is intentionally small so that other packages in the ctpu tool can be effectively tested.
type ResourceManagementCP struct {
	config  *config.Config
	service *cloudresourcemanager.Service
}

func newResourceManagementCP(config *config.Config, client *http.Client, ctpuVersion string) (*ResourceManagementCP, error) {
	service, err := cloudresourcemanager.New(client)
	if err != nil {
		return nil, err
	}
	service.UserAgent = "ctpu/" + ctpuVersion
	return &ResourceManagementCP{
		config:  config,
		service: service,
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
		for _, member := range binding.Members {
			if member == tpuMemberStr {
				// The TPU service account has already been enabled. Make no modifications.
				return nil
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

// AddTPUUserAgent adds the TPU user agent to enable GCS access and send logging
//
// It is a no-op if the tpuUserAgent has already been granted some access.
func (r *ResourceManagementCP) AddTPUUserAgent(tpuUserAgent string) error {
	req := cloudresourcemanager.GetIamPolicyRequest{}
	policy, err := r.service.Projects.GetIamPolicy(r.config.Project, &req).Do()
	if err != nil {
		return err
	}
	policy = r.addAgentToPolicy(tpuUserAgent, policy)
	if policy != nil {
		log.Printf("Updating the project's IAM policy to add the TPU service account ('%s') to roles: %s and %s.", tpuUserAgent, loggingRole, storageRole)
		req := cloudresourcemanager.SetIamPolicyRequest{Policy: policy}
		if _, err := r.service.Projects.SetIamPolicy(r.config.Project, &req).Do(); err != nil {
			return err
		}
	}
	return nil
}
