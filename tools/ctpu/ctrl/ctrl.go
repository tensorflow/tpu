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

// Package ctrl contains simplified abstractions for interacting with Cloud APIs.
//
// These simplified APIs are useful for testing the workflows executed by the
// commands package.
package ctrl

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"os"

	"context"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/compute/v1"
)

// Ctrl contains the set of Control Plane APIs required to manage Cloud TPU flocks.
type Ctrl struct {
	client             *http.Client
	GCE                *GCECP
	TPU                *TPUCP
	CLI                *GCloudCLI
	ResourceManagement *ResourceManagementCP
}

type loggingRoundTripper struct {
	underlying http.RoundTripper
}

func (l *loggingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	reqBytes, err := httputil.DumpRequestOut(req, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error dumping request: %#v", req)
		return nil, err
	}
	resp, err := l.underlying.RoundTrip(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, `=========================================
-------- Request ---------
%s
---------- Error ---------
%v
=========================================
`, reqBytes, err)
		return resp, err
	}
	respBytes, err := httputil.DumpResponse(resp, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error dumping response; request: %q, response err: %#v", reqBytes, err)
		return resp, nil
	}

	fmt.Fprintf(os.Stderr, `=========================================
-------- Request ---------
%s
-------- Response --------
%s
=========================================
`, reqBytes, respBytes)
	return resp, nil
}

// New creates a new Ctrl instance with fully populated control plane values.
func New(ctx context.Context, config config.Config, ctpuVersion string, logRequests bool) (*Ctrl, error) {

	// TODO(saeta): Add TPU scopes.
	client, err := google.DefaultClient(ctx, compute.ComputeScope)

	if logRequests {
		client.Transport = &loggingRoundTripper{underlying: client.Transport}
	}

	if err != nil {
		return nil, err
	}

	serviceMgmt, err := newServiceManagementCP(config, client, ctpuVersion)
	if err != nil {
		return nil, err
	}

	gce, err := newGCECP(config, client, serviceMgmt, ctpuVersion)
	if err != nil {
		return nil, err
	}

	resourceMgmt, err := newResourceManagementCP(config, client, ctpuVersion)
	if err != nil {
		return nil, err
	}

	tpu, err := newTPUCP(config, client, serviceMgmt, ctpuVersion)
	if err != nil {
		return nil, err
	}

	cli := &GCloudCLI{config}

	return &Ctrl{
		client:             client,
		GCE:                gce,
		TPU:                tpu,
		CLI:                cli,
		ResourceManagement: resourceMgmt,
	}, nil
}
