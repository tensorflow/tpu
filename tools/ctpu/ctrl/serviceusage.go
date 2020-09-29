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
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/serviceusage/v1"
)

type serviceUsageCP struct {
	services   *serviceusage.ServicesService
	operations *serviceusage.OperationsService
	config     *config.Config
}

func newServiceUsageCP(config *config.Config, client *http.Client, userAgent string) (*serviceUsageCP, error) {
	apiService, err := serviceusage.New(client)
	if err != nil {
		return nil, err
	}
	apiService.UserAgent = userAgent
	return &serviceUsageCP{
		services:   apiService.Services,
		operations: apiService.Operations,
		config:     config,
	}, nil
}

func (s *serviceUsageCP) checkIfEnabled(serviceName string) (bool, error) {
	resp, err := s.services.Get(fmt.Sprintf("projects/%s/services/%s", s.config.Project, serviceName)).Do()
	if err != nil {
		return false, err
	}
	return resp.State == "ENABLED", nil
}

func (s *serviceUsageCP) pollUntilOperationComplete(serviceName string, operation *serviceusage.Operation) error {
	if operation.Error != nil {
		return errors.New(operation.Error.Message)
	}
	if operation.Done {
		return nil
	}
	for {
		time.Sleep(5 * time.Second) // Poll every 5 seconds
		op, err := s.operations.Get(operation.Name).Do()
		if err != nil {
			return err
		}
		if op.Error != nil {
			return fmt.Errorf("error enabling '%s' service: %#v", serviceName, op)
		}
		if op.Done {
			return nil
		}
	}
}

func (s *serviceUsageCP) enableService(serviceName string) error {
	operation, err := s.services.Enable(
		fmt.Sprintf("projects/%s/services/%s", s.config.Project, serviceName),
		&serviceusage.EnableServiceRequest{}).Do()
	if err != nil {
		return err
	}
	return s.pollUntilOperationComplete(serviceName, operation)
}
