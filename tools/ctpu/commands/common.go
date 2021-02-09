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

// Package commands contains commands available to ctpu users.
package commands

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"flag"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
	"golang.org/x/sync/errgroup"
)

type tpuCmd struct {
	dryRun           bool
	tpuOnly          bool
	skipWaiting      bool
	skipConfirmation bool
}

func (c *tpuCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&c.dryRun, "dry-run", false,
		"Do not make changes; print only what would have happened.")
	f.BoolVar(&c.tpuOnly, "tpu-only", false,
		"Do not modify the Compute Engine VM, only modify the TPU (useful if you want to edit code on the VM without paying for the TPU).")
	f.BoolVar(&c.skipWaiting, "nowait", false,
		"Don't wait for asynchronous operations to complete (e.g. TPU deletion, Compute Engine VM halting)")
	f.BoolVar(&c.skipConfirmation, "noconf", false, "Skip confirmation about deleting resources.")
}

func (tpuCmd) printConfig(cfg *config.Config) {
	fmt.Printf(`ctpu will use the following configuration values:
	Name:          %s
	Zone:          %s
	GCP Project:   %s
`, cfg.FlockName, cfg.Zone, cfg.Project)
}

type tpuInstanceFn func() (*ctrl.TPUInstance, error)
type gceInstanceFn func() (*ctrl.GCEInstance, error)
type cpCommand func() (ctrl.LongRunningOperation, error)

func cleanUpVM(cfg *config.Config, gceCP gceInstanceFn, vmCommand cpCommand, dryRun bool, actionName string, requiresRunning bool) (ctrl.LongRunningOperation, error) {
	vm, err := gceCP()
	if err != nil {
		return nil, err
	}
	if vm == nil {
		return nil, fmt.Errorf("no Compute Engine VM %q found", cfg.FlockName)
	}
	if !vm.IsRunning() && requiresRunning {
		log.Printf("Compute Engine VM %s not running.\n", cfg.FlockName)
		return nil, nil
	}

	log.Printf("%s Compute Engine VM %q...\n", actionName, cfg.FlockName)
	if !dryRun {
		return vmCommand()
	}
	return nil, nil
}

func cleanUpTPU(cfg *config.Config, tpuCP tpuInstanceFn, tpuCommand cpCommand, dryRun bool) (ctrl.LongRunningOperation, error) {
	tpu, err := tpuCP()
	if err != nil {
		return nil, err
	}
	if tpu == nil {
		return nil, fmt.Errorf("no TPU %s found", cfg.FlockName)
	}

	log.Printf("Deleting TPU %s...\n", cfg.FlockName)
	if !dryRun {
		return tpuCommand()
	}
	return nil, nil
}

func waitForLongRunningOperations(operation string, skipWaiting bool, gceOp, tpuOp ctrl.LongRunningOperation) error {
	if skipWaiting {
		fmt.Printf(`All %s operations have been initiated successfully.
	Exiting early due to the --nowait flag.
`, operation)
		return nil
	}

	fmt.Printf(`All %q operations have been initiated successfully. They will
run to completion even if you kill ctpu (e.g. by pressing Ctrl-C). When the
operations have finished running, ctpu will exit. If you would like your shell
back, you can press Ctrl-C now. Note: Next time you run ctpu, you can pass the
--nowait flag to get your shell back immediately.
`, operation)

	var g errgroup.Group

	if gceOp != nil {
		g.Go(func() error {
			return gceOp.LoopUntilComplete()
		})
	}
	if tpuOp != nil {
		g.Go(func() error {
			return tpuOp.LoopUntilComplete()
		})
	}
	err := g.Wait()
	if err == nil {
		fmt.Printf("ctpu %s completed successfully.\n", operation)
	}
	return err
}

var versionRegex = regexp.MustCompile("^(\\d+)\\.(\\d+)(.*)$")
var nightlyVersionRegex = regexp.MustCompile("^nightly(.*)$")
var patchNumberRegex = regexp.MustCompile("^\\.(\\d+)$")

// Expected versions look like one of the following formats:
//  - 1.6
//  - 1.7-RC3
//  - nightly
//  - nightly-20180218
type parsedVersion struct {
	Major     int
	Minor     int
	Patch     int
	IsNightly bool
	Modifier  string
}

func (p parsedVersion) isUnknown() bool {
	return p.Major == 0 && p.Minor == 0 && !p.IsNightly
}

func (p parsedVersion) versionString() string {
	if p.IsNightly {
		return fmt.Sprintf("nightly%s", p.Modifier)
	}
	if p.Major == 0 && p.Minor == 0 && p.Patch == 0 {
		return p.Modifier
	}

	// From TF 2.4 onwards, image name uses patch format by default. But
	// before that the image name can be only <major>.<minor>. e.g.:
	// `2.4.0`, `2.4.1` and `2.3` all are valid versions.
	if p.Patch != 0 || p.Major >= 2 && p.Minor >= 4 {
		return fmt.Sprintf("%d.%d.%d%s", p.Major, p.Minor, p.Patch, p.Modifier)
	}

	return fmt.Sprintf("%d.%d%s", p.Major, p.Minor, p.Modifier)
}

// Defines a useful sort of []parsedVersion.
//
// Sorts stable versions in descending order, followed by nightly, followed by modified nightlies
type sortedParsedVersions []parsedVersion

func (s sortedParsedVersions) Len() int      { return len(s) }
func (s sortedParsedVersions) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s sortedParsedVersions) Less(i, j int) bool {
	// Both non-nightlies, non-unknowns
	if !s[i].IsNightly && !s[j].IsNightly && !s[i].isUnknown() && !s[j].isUnknown() {
		if s[i].Major != s[j].Major {
			return s[i].Major > s[j].Major
		}
		if s[i].Minor != s[j].Minor {
			return s[i].Minor > s[j].Minor
		}
		if s[i].Patch != s[j].Patch {
			return s[i].Patch > s[j].Patch
		}
		if len(s[i].Modifier) == 0 {
			return true
		}
		if len(s[j].Modifier) == 0 {
			return false
		}
		return strings.Compare(s[i].Modifier, s[j].Modifier) >= 0
	}
	// Both nightlies
	if s[i].IsNightly && s[j].IsNightly {
		if len(s[i].Modifier) == 0 {
			return true
		}
		if len(s[j].Modifier) == 0 {
			return false
		}
		return strings.Compare(s[i].Modifier, s[j].Modifier) >= 0
	}
	// Both unknown versions
	if s[i].isUnknown() && s[j].isUnknown() {
		return strings.Compare(s[i].Modifier, s[j].Modifier) < 0 // Alpha sort!
	}
	// If one is an unknown version, sort after
	if s[i].isUnknown() {
		return false
	}
	if s[j].isUnknown() {
		return true
	}
	// One is nightly, one is not.
	if s[i].IsNightly {
		return false
	}
	return true
}

// Note: this destructively sorts s.
func (s sortedParsedVersions) LatestStableRelease() (string, error) {
	sort.Sort(s)
	for _, version := range s {
		if version.IsNightly {
			return "", fmt.Errorf("no stable releases found")
		}
		if len(version.Modifier) == 0 {
			return version.versionString(), nil
		}
	}
	return "", fmt.Errorf("no stable releases found")
}

func parseVersion(version string) (parsedVersion, error) {
	versionMatch := versionRegex.FindStringSubmatch(version)
	nightlyMatch := nightlyVersionRegex.FindStringSubmatch(version)

	if len(versionMatch) == 0 && len(nightlyMatch) == 0 {
		return parsedVersion{Modifier: version}, nil
	}

	if len(versionMatch) != 0 && len(nightlyMatch) != 0 {
		return parsedVersion{}, fmt.Errorf("internal version error: bad TF version: '%s'", version)
	}

	if len(versionMatch) > 0 {
		if len(versionMatch) < 3 {
			return parsedVersion{}, fmt.Errorf("could not parse version %q", version)
		}
		major, err := strconv.Atoi(versionMatch[1])
		if err != nil {
			return parsedVersion{}, fmt.Errorf("could not parse major version %q: %v", version, err)
		}
		minor, err := strconv.Atoi(versionMatch[2])
		if err != nil {
			return parsedVersion{}, fmt.Errorf("could not parse minor version %q: %v", version, err)
		}
		parsed := parsedVersion{
			Major: major,
			Minor: minor,
		}
		if len(versionMatch) == 4 {
			parsed.Patch = 0
			patchMatch := patchNumberRegex.FindStringSubmatch(versionMatch[3])
			if len(patchMatch) != 0 {
				patch, err := strconv.Atoi(patchMatch[1])
				if err == nil {
					parsed.Patch = patch
				}
			} else {
				parsed.Modifier = versionMatch[3]
			}
		}
		return parsed, nil
	}

	if len(nightlyMatch) > 0 {
		parsed := parsedVersion{
			IsNightly: true,
		}

		if len(nightlyMatch) == 2 {
			parsed.Modifier = nightlyMatch[1]
		}
		return parsed, nil
	}

	return parsedVersion{}, fmt.Errorf("unreachable code reached in parseVersion while parsing version %q", version)
}

func askForConfirmation(prompt string) (bool, error) {
	reader := bufio.NewReader(os.Stdin)

	fmt.Printf("%s [Yn]: ", prompt)

	response, err := reader.ReadString('\n')
	if err != nil {
		return false, err
	}

	canon := strings.ToLower(strings.TrimSpace(response))

	if len(canon) == 0 || canon[0] == 'y' {
		return true, nil
	}

	if canon[0] == 'n' {
		return false, nil
	}

	return false, fmt.Errorf("unexpected response: %q", strings.TrimSpace(response))
}
