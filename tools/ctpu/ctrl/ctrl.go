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
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/tensorflow/tpu/tools/ctpu/config"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/tpu/v1alpha1"
)

// LongRunningOperation represents asynchronous control plane operations.
type LongRunningOperation interface {
	// LoopUntilComplete pools the control plane until the operation is complete.
	LoopUntilComplete() error
}

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
func New(ctx context.Context, config *config.Config, ctpuVersion string, logRequests bool) (*Ctrl, error) {

	var client *http.Client
	var err error
	if config.Environment == "devshell" {
		ts := &devshellTokenSource{}
		firstToken, err := ts.Token()
		if err != nil {
			return nil, err
		}
		client = oauth2.NewClient(ctx, oauth2.ReuseTokenSource(firstToken, ts))
	} else {
		client, err = google.DefaultClient(ctx, compute.ComputeScope, tpu.CloudPlatformScope)
		if err != nil {
			return nil, err
		}
	}

	if logRequests {
		client.Transport = &loggingRoundTripper{underlying: client.Transport}
	}

	userAgent := fmt.Sprintf("ctpu/%s env/%s", ctpuVersion, config.Environment)

	serviceMgmt, err := newServiceManagementCP(config, client, userAgent)
	if err != nil {
		return nil, err
	}

	gce, err := newGCECP(config, client, serviceMgmt, userAgent)
	if err != nil {
		return nil, err
	}

	resourceMgmt, err := newResourceManagementCP(ctx, config, client, userAgent)
	if err != nil {
		return nil, err
	}

	tpu, err := newTPUCP(config, client, serviceMgmt, userAgent)
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

type devshellTokenSource struct {
}

func devshellPort() (int, error) {
	for _, e := range os.Environ() {
		pair := strings.Split(e, "=")
		if pair[0] == "DEVSHELL_CLIENT_PORT" {
			if len(pair) != 2 {
				return 0, fmt.Errorf("devshell port: unexpected environment value: %q", e)
			}
			return strconv.Atoi(pair[1])

		}
	}
	return 0, fmt.Errorf("devshell port: environment variable DEVSHELL_CLIENT_PORT not found")
}

func (d *devshellTokenSource) parseResponse(response string) (*oauth2.Token, error) {
	items := make([]interface{}, 0)
	err := json.Unmarshal([]byte(response), &items)
	if err != nil {
		return nil, err
	}
	if len(items) < 4 {
		return nil, fmt.Errorf("devshell token: too few fields found parsing token server response %q", response)
	}
	token := oauth2.Token{}
	accessToken, ok := items[2].(string)
	if !ok {
		return nil, fmt.Errorf("devshell token: access token not parsed as string %q", response)
	}
	token.AccessToken = accessToken

	expiryDelta, ok := items[3].(float64)
	if !ok {
		log.Printf("Warning: could not parse expiry time for token: %q", response)
	}

	parsedExpiry, err := time.ParseDuration(fmt.Sprintf("%fs", expiryDelta))
	if err != nil {
		log.Printf("Warning: could not parse expiry time for token: %q", response)
	} else {
		token.Expiry = time.Now().Add(parsedExpiry)
	}

	return &token, nil
}

func (d *devshellTokenSource) Token() (*oauth2.Token, error) {

	port, err := devshellPort()
	if err != nil {
		return nil, err
	}
	conn, err := net.Dial("tcp", fmt.Sprintf("localhost:%d", port))
	if err != nil {
		return nil, err
	}

	// Send request.
	requestString := "2\n[]"
	bytesSent, err := fmt.Fprint(conn, requestString)
	if err != nil {
		return nil, err
	}
	if bytesSent != len(requestString) {
		return nil, fmt.Errorf("devshell token: full request not sent")
	}

	reader := bufio.NewReader(conn)
	resp1, err := reader.ReadString('\n')
	if err != nil {
		return nil, err
	}
	respBytes, err := strconv.Atoi(strings.TrimSpace(resp1))
	if err != nil {
		return nil, fmt.Errorf("devshell token: response not valid: %q", resp1)
	}

	resp := make([]byte, respBytes)
	if _, err := io.ReadFull(reader, resp); err != nil {
		return nil, err
	}

	return d.parseResponse(string(resp))
}
