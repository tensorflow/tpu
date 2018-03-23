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

// Package config contains common configuration to all commands & control abstractions.
package config

import (
	"errors"
	"fmt"
	"log"
	"os/user"
	"regexp"

	"flag"
)

// Config encapsulates all common configuration required to execute a command.
type Config interface {
	// Environment is the current environment ctpu is running in
	Environment() string
	// The current active configuration we are using as default.
	ActiveConfiguration() string
	// The "flock" name to use when creating the VMs and TPUs
	FlockName() string
	// The GCP project we will use to allocate / deallocate TPUs and VMs.
	Project() string
	// The GCE Zone we will use to allocate / deallocate TPUs and VMs.
	Zone() string

	// TODO(saeta): Add additional parameters such as:
	//   - VPC network
	//   - TF version
	//   - machine size
	//   - quiet config?
}

const defaultZone = "us-central1-c"

// Flag overrides
var (
	flockOverride, projectOverride, zoneOverride string
)

// RegisterFlags registers the flags with the flags package.
//
// It should be called from the main() function before parsing flags.
func RegisterFlags() {
	flag.StringVar(&flockOverride, "name", "",
		"Override the name to use for VMs and TPUs (defaults to your username).")
	flag.StringVar(&projectOverride, "project", "",
		`Override the GCP project name to use when allocating VMs and TPUs.
       By default, it picks a reasonable value from either your gcloud
       configuration, or the GCE metadata. If a good value cannot be found, you
       will be required to provide a value on the command line.)`)
	flag.StringVar(&zoneOverride, "zone", "",
		`Override the GCE zone to use when allocating & deallocating resources.
        By default, it picks a reasonable value from either your gcloud
        configuration, or the GCE metadata. If a good value cannot be found, you
        will be required to provide a value on the command line.)`)
}

// Validation regular expression.
var usernameRegex = regexp.MustCompile("^([A-z0-9-_]+)")
var flocknameRegex = regexp.MustCompile("^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")

// NewConfig constructs a Config object from the environment.
func NewConfig() (Config, error) {
	env, err := buildEnvConfig()
	if err != nil {
		return nil, err
	}
	config, err := newFlagOverrideConfig(env)
	if err != nil {
		return nil, err
	}
	return config, nil
}

type flagOverrideConfig struct {
	e         *envConfig
	flockName string
}

func newFlagOverrideConfig(e *envConfig) (*flagOverrideConfig, error) {
	flockName, err := computeFlockName(e)
	if err != nil {
		return nil, err
	}
	cfg := flagOverrideConfig{
		e:         e,
		flockName: flockName,
	}
	err = cfg.validate()
	if err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (f *flagOverrideConfig) validate() error {
	if len(f.FlockName()) == 0 {
		return errors.New("no flock name specified, please set one on the command line --name=NAME")
	}
	if err := checkFlockName(f.FlockName()); err != nil {
		return err
	}
	if len(f.Project()) == 0 {
		return errors.New("no project specified, please set one on the command line --project $PROJECT_NAME")
	}
	if len(f.Zone()) == 0 {
		zoneOverride = defaultZone
		log.Printf("WARNING: Setting zone to %q", defaultZone)
	}
	return nil
}

func computeFlockName(e *envConfig) (flockName string, err error) {
	if len(flockOverride) > 0 {
		flockName = flockOverride
	} else {
		if len(e.account) < 2 {
			curUser, err := user.Current()
			if err != nil {
				return "", err
			}
			flockName = curUser.Username
		} else {
			submatches := usernameRegex.FindStringSubmatch(e.account)
			if len(submatches) != 2 {
				return "", fmt.Errorf("could not determine a flock name based on the current user account (%q)", e.account)
			}
			flockName = submatches[1]
		}
	}
	if len(flockName) < 2 {
		return "", errors.New("could not compute a name for your cluster; please specify one using the --name parameter on the command line")
	}
	return
}

func checkFlockName(name string) error {
	if !flocknameRegex.MatchString(name) {
		return fmt.Errorf("flock name %q is not a valid flock name (must match regex: %q)", name, flocknameRegex.String())
	}
	return nil
}

func (f *flagOverrideConfig) ActiveConfiguration() string {
	return f.e.activeConfiguration
}

func (f *flagOverrideConfig) FlockName() string {
	return f.flockName
}

func (f *flagOverrideConfig) Project() string {
	if len(projectOverride) > 0 {
		return projectOverride
	}
	return f.e.project
}

func (f *flagOverrideConfig) Zone() string {
	if len(zoneOverride) > 0 {
		return zoneOverride
	}
	return f.e.zone
}

func (f *flagOverrideConfig) Environment() string {
	return f.e.environment
}

// TestConfig is a `Config` that can be easily used in tests.
type TestConfig struct {
	ActiveConfigurationVal string
	FlockNameVal           string
	ProjectVal             string
	ZoneVal                string
	EnvironmentVal         string
}

// ActiveConfiguration returns the active configuration
func (t *TestConfig) ActiveConfiguration() string {
	return t.ActiveConfigurationVal
}

// FlockName returns the flock name
func (t *TestConfig) FlockName() string {
	return t.FlockNameVal
}

// Project returns the GCP project to use
func (t *TestConfig) Project() string {
	return t.ProjectVal
}

// Zone returns the GCE zone to use
func (t *TestConfig) Zone() string {
	return t.ZoneVal
}

// Environment returns the current environment
func (t *TestConfig) Environment() string {
	if t.EnvironmentVal == "" {
		return "gcloud"
	}
	return t.EnvironmentVal
}
