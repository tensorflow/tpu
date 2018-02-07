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
	"google.golang.org/api/servicemanagement/v1"
)

type serviceManagementCP struct {
	services   *servicemanagement.ServicesService
	operations *servicemanagement.OperationsService
	config     config.Config
}

func newServiceManagementCP(config config.Config, client *http.Client, ctpuVersion string) (*serviceManagementCP, error) {
	apiService, err := servicemanagement.New(client)
	if err != nil {
		return nil, err
	}
	apiService.UserAgent = "ctpu/" + ctpuVersion
	return &serviceManagementCP{
		services:   apiService.Services,
		operations: apiService.Operations,
		config:     config,
	}, nil
}

func (s *serviceManagementCP) checkIfEnabled(serviceName string) (bool, error) {
	response, err := s.services.List().ConsumerId(s.consumerID()).Do()
	if err != nil {
		return false, err
	}
	for _, managedService := range response.Services {
		if managedService.ServiceName == serviceName {
			return true, nil
		}
	}
	// TODO: handle additional pages in the response.
	return false, nil
}

func (s *serviceManagementCP) pollUntilOperationComplete(serviceName string, operation *servicemanagement.Operation) error {
	if operation.Error != nil {
		return errors.New(operation.Error.Message)
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

func (s *serviceManagementCP) consumerID() string {
	return fmt.Sprintf("project:%s", s.config.Project())
}

func (s *serviceManagementCP) enableService(serviceName string) error {
	req := servicemanagement.EnableServiceRequest{
		ConsumerId: s.consumerID(),
	}

	operation, err := s.services.Enable(serviceName, &req).Do()
	if err != nil {
		return err
	}
	return s.pollUntilOperationComplete(serviceName, operation)
}
