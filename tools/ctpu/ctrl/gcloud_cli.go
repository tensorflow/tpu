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
	"os"
	"os/exec"
	"syscall"

	"github.com/tensorflow/tpu/tools/ctpu/config"
)

// GCloudCLI abstracts away interacting with a locally installed GCloud to facilitate testing
type GCloudCLI struct {
	Config *config.Config
}

// IsGcloudInstalled returnes true if the gcloud cli is installed, false otherwise.
func (GCloudCLI) IsGcloudInstalled() bool {
	_, err := exec.LookPath("gcloud")
	return err == nil
}

// PrintInstallInstructions prints instructions for how to install gcloud to the console.
func (GCloudCLI) PrintInstallInstructions() {
	fmt.Println(`Error: gcloud does not appear to be installed. Please see https://cloud.google.com/sdk/downloads for instructions on how to install the gcloud CLI.
If you recently installed gcloud and you're still encountering this error, ensure your path has been set correctly and try running the ctpu tool in a fresh shell.`)
}

func (g GCloudCLI) makeExecCommand(forwardPorts, forwardAgent bool, tpuInstance *TPUInstance) []string {
	command := []string{
		"gcloud",
		"compute",
		"ssh",
		"--zone",
		g.Config.Zone,
		"--project",
		g.Config.Project,
		g.Config.FlockName,
	}
	command = append(command, "--", "-o", "SendEnv=TPU_NAME") // Always send the TPU_NAME environment variable.
	if forwardAgent && g.Config.Environment != "devshell" {
		command = append(command, "-A")
	}
	if forwardPorts {
		if g.Config.Environment == "devshell" {
			command = append(command, "-L", "8080:localhost:6006", "-L", "8081:localhost:8888")
			command = append(command, "-y") // Suppresses channel errors from ssh.
		} else {
			command = append(command, "-L", "6006:localhost:6006", "-L", "8888:localhost:8888")
			if tpuInstance != nil {
				command = append(command, "-L", "8470:"+tpuInstance.NetworkEndpoints[0].IpAddress+":8470", "-L", "8466:"+tpuInstance.NetworkEndpoints[0].IpAddress+":8466")
			}
		}
	}
	return command
}

// MakeEnviron creates an environment that includes the extra environment variable TPU_NAME
//
// This new environment is then suitable for use when calling execve to execute the gcloud tool.
func (g GCloudCLI) MakeEnviron() []string {
	env := os.Environ()
	env = append(env, fmt.Sprintf("TPU_NAME=%s", g.Config.FlockName))
	return env
}

// SSHToInstance opens an ssh connection to the Compute Engine VM in the flock.
//
// If an error is encountered, an error is returned.
//
// Note: SSHToInstance calls syscall.Exec which replaces the contents of the current process with
// the ssh command (via the gcloud helper tool). As a result in the successful case, SSHToInstance
// never returns.
func (g GCloudCLI) SSHToInstance(forwardPorts, forwardAgent bool, tpuInstance *TPUInstance) error {
	path, err := exec.LookPath("gcloud")
	if err != nil {
		return err
	}
	return syscall.Exec(path, g.makeExecCommand(forwardPorts, forwardAgent, tpuInstance), g.MakeEnviron())
}
