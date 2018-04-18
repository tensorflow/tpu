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
	"strings"

	"flag"
)

const defaultZone = "us-central1-c"

// Validation regular expression.
var usernameRegex = regexp.MustCompile("^([A-z0-9-_]+)")
var flocknameRegex = regexp.MustCompile("^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
var invalidChars = regexp.MustCompile("([^-a-z0-9])")
var dupedDash = regexp.MustCompile("(--+)")

// Config encapsulates all common configuration required to execute a command.
type Config struct {
	// Environment is the current environment ctpu is running in
	Environment string
	// The current active configuration we are using as default.
	ActiveConfiguration string
	// The "flock" name to use when creating the VMs and TPUs
	FlockName string
	// The GCP project we will use to allocate / deallocate TPUs and VMs.
	Project string
	// The GCE Zone we will use to allocate / deallocate TPUs and VMs.
	Zone string

	// The following fields are maintained for internal use only
	account string
}

// SetFlags registers flag overrides for the config object.
func (c *Config) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.FlockName, "name", c.FlockName, "Override the name to use for VMs and TPUs (defaults to your username).")
	f.StringVar(&c.Project, "project", c.Project,
		`Override the GCP project name to use when allocating VMs and TPUs.
       By default, ctpu picks a reasonable value from either your gcloud
       configuration, or the GCE metadata. If a good value cannot be found, you
       will be required to provide a value on the command line.)`)
	f.StringVar(&c.Zone, "zone", "",
		`Override the GCE zone to use when allocating & deallocating resources.
        By default, it picks a reasonable value from either your gcloud
        configuration, or the GCE metadata. If a good value cannot be found, you
        will be required to provide a value on the command line.)`)
}

// Validate verifies that the configuration has been fully populated.
func (c *Config) Validate() error {
	err := checkFlockName(c.FlockName)
	if err != nil {
		return err
	}
	if c.Project == "" {
		return errors.New("no project specified, please set one on the command line --project $PROJECT_NAME")
	}
	if c.Zone == "" {
		c.Zone = defaultZone
		log.Printf("WARNING: Setting zone to %q", defaultZone)
	}
	return nil
}

// FromEnv constructs a Config object from the environment.
func FromEnv() (cfg *Config, err error) {
	if isDevshell() {
		cfg, err = devshellConfig()
	} else {
		cfg, err = gcloudConfig()
	}
	if err != nil {
		return nil, err
	}
	cfg.computeFlockName()
	return cfg, nil
}

func cleanFlockName(flockName string) string {
	flockName = invalidChars.ReplaceAllString(flockName, "-")
	flockName = dupedDash.ReplaceAllString(flockName, "-")
	flockName = strings.Trim(flockName, "-")
	return flockName
}

func (c *Config) computeFlockName() {
	if len(c.account) < 2 {
		curUser, err := user.Current()
		if err != nil {
			return
		}
		c.FlockName = cleanFlockName(curUser.Username)
	} else {
		submatches := usernameRegex.FindStringSubmatch(c.account)
		if len(submatches) != 2 {
			return
		}
		if len(submatches[1]) >= 2 {
			c.FlockName = cleanFlockName(submatches[1])
		}
	}
}

func checkFlockName(name string) error {
	if len(name) < 2 {
		return fmt.Errorf("flock name %q is not a valid flock name (must be at least 2 characters)", name)
	}
	if !flocknameRegex.MatchString(name) {
		return fmt.Errorf("flock name %q is not a valid flock name (must match regex: %q)", name, flocknameRegex.String())
	}
	return nil
}
