// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

package commands

import (
	"context"
	"fmt"
	"log"
	"sort"

	"cloud.google.com/go/storage"
	"flag"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/cloudresourcemanager/v1beta1"
)

const loggingRole = "roles/logging.logWriter"
const storageRole = "roles/storage.admin" // Note storage.objectAdmin does not work in certain cases, and thus we need roles/storage.admin.
const bigtableRole = "roles/bigtable.user"
const bigtableReadonlyRole = "roles/bigtable.reader"
const tpuServiceAgent = "roles/tpu.serviceAgent"
const serviceAgentFormatString = "serviceAccount:service-%d@cloud-tpu.iam.gserviceaccount.com"
const roleWriter = "WRITER"

// AuthResourceManagementCP abstracts the key operations the auth subcommand family must perform.
type AuthResourceManagementCP interface {
	GetProject() (*cloudresourcemanager.Project, error)

	// GetProjectPolicy retrieves the current project's IAM policy.
	GetProjectPolicy() (*cloudresourcemanager.Policy, error)

	// SetProjectPolicy sets the policy.
	SetProjectPolicy(policy *cloudresourcemanager.Policy) error

	// GetBucketACL retrieves a Cloud Storage bucket's ACL list.
	GetBucketACL(ctx context.Context, bucket string) ([]storage.ACLRule, error)

	// SetBucketACL adds the entity + role pair to the ACL list for the provided bucket.
	SetBucketACL(ctx context.Context, bucket string, entity storage.ACLEntity, role storage.ACLRole) error
}

type authCmd struct {
	cfg *config.Config
	cp  AuthResourceManagementCP
}

// AuthCommand constructs the auth subcommand tree.
func AuthCommand(cfg *config.Config, cp AuthResourceManagementCP) subcommands.Command {
	return &authCmd{cfg: cfg, cp: cp}
}

func (authCmd) Name() string {
	return "auth"
}

func (c *authCmd) SetFlags(f *flag.FlagSet) {
	c.cfg.SetFlags(f)
}

func (authCmd) Synopsis() string {
	return "subcommands relating to configuring authorization for Cloud TPUs"
}

func (authCmd) Usage() string {
	return "ctpu auth [subcommand]\n"
}

func (c *authCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	cmd := subcommands.NewCommander(flags, "ctpu auth")
	cmd.Register(&authListCmd{c.cfg, c.cp}, "")
	cmd.Register(&authAddBigtable{c.cfg, c.cp, false, false}, "")
	cmd.Register(&authAddGcs{c.cfg, c.cp, false, false}, "")

	cmd.Register(cmd.FlagsCommand(), "usage")
	cmd.Register(cmd.HelpCommand(), "usage")
	cmd.Register(cmd.CommandsCommand(), "usage")

	return cmd.Execute(ctx)
}

type authListCmd struct {
	cfg *config.Config
	cp  AuthResourceManagementCP
}

func (authListCmd) Name() string {
	return "list"
}

func (authListCmd) Synopsis() string {
	return "displays Cloud TPU service account authorizations"
}

func (authListCmd) Usage() string {
	return `ctpu auth list

This subcommand will retrieve and display the project-level permissions for
the Cloud TPU service account.

Note: it does not list Cloud Storage bucket-level permissions.

`
}

func (c *authListCmd) SetFlags(f *flag.FlagSet) {
	c.cfg.SetFlags(f)
}

func (c *authListCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	project, err := c.cp.GetProject()
	if err != nil {
		log.Fatalf("Error retrieving project metadata for project %q: %v", c.cfg.Project, err)
	}
	sa := makeTPUServiceName(project)
	policy, err := c.cp.GetProjectPolicy()
	if err != nil {
		log.Fatalf("Error retrieving project IAM policy for project %q: %v", c.cfg.Project, err)
	}
	bindings := filterBindings(sa, policy)
	var prettyRoleNames []string
	for _, b := range bindings {
		if b.Role == bigtableRole {
			prettyRoleNames = append(prettyRoleNames, "Cloud Bigtable")
		} else if b.Role == bigtableReadonlyRole {
			prettyRoleNames = append(prettyRoleNames, "Cloud Bigtable (read only)")
		} else if b.Role == storageRole {
			prettyRoleNames = append(prettyRoleNames, "Cloud Storage")
		} else if b.Role == loggingRole {
			prettyRoleNames = append(prettyRoleNames, "Logging")
		} else if b.Role == tpuServiceAgent {
			prettyRoleNames = append(prettyRoleNames, "TPU Permissions (required)")
		} else {
			prettyRoleNames = append(prettyRoleNames, b.Role)
		}
	}
	sort.Strings(prettyRoleNames)
	fmt.Printf("The Cloud TPU service account for project %q has the following permissions:\n", project.ProjectId)
	for _, n := range prettyRoleNames {
		fmt.Printf("\t%s\n", n)
	}
	return subcommands.ExitSuccess
}

type authAddBigtable struct {
	cfg      *config.Config
	cp       AuthResourceManagementCP
	readonly bool
	skipConf bool
}

func (authAddBigtable) Name() string {
	return "add-bigtable"
}

func (authAddBigtable) Synopsis() string {
	return "ensures Cloud TPUs are authorized for Cloud Bigtable"
}

func (authAddBigtable) Usage() string {
	return "ctpu auth add-bigtable\n"
}

func (c *authAddBigtable) SetFlags(f *flag.FlagSet) {
	c.cfg.SetFlags(f)
	f.BoolVar(&c.readonly, "readonly", false, "If modifying IAM policy, use read-only permissions instead of read-write permissions.")
	f.BoolVar(&c.skipConf, "skip-confirmation", false, "Skip confirmation before changes are made.")
}

func (c *authAddBigtable) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	project, err := c.cp.GetProject()
	if err != nil {
		log.Fatalf("Error retrieving project metadata for project %q: %v", c.cfg.Project, err)
	}
	sa := makeTPUServiceName(project)
	policy, err := c.cp.GetProjectPolicy()
	if err != nil {
		log.Fatalf("Error retrieving project IAM policy for project: %q: %v", c.cfg.Project, err)
	}

	var bindingToModify *cloudresourcemanager.Binding
	for _, b := range policy.Bindings {
		if b.Role == bigtableRole {
			if bindingHasServiceAccount(sa, b) {
				fmt.Printf("Cloud TPUs in project %q already have read/write access to Cloud Bigtable.\nNo changes were made.\n", c.cfg.Project)
				return subcommands.ExitSuccess
			}
			if !c.readonly || bindingToModify == nil {
				bindingToModify = b
			}
		}
		if b.Role == bigtableReadonlyRole && c.readonly {
			if bindingHasServiceAccount(sa, b) {
				fmt.Printf("Cloud TPUs in project %q already have read-only access to Cloud Bigtable.\nNo changes were made.\n", c.cfg.Project)
				return subcommands.ExitSuccess
			}
			bindingToModify = b
		}
	}
	if !c.skipConf {
		ok, err := askForConfirmation(fmt.Sprintf("Is it okay to modify IAM permissions for project %q to grant Cloud TPUs access to Cloud Bigtable?", c.cfg.Project))
		if err != nil {
			log.Fatalf("Error while asking for confirmation: %v", err)
		}
		if !ok {
			fmt.Printf("No changes made.\n")
			return subcommands.ExitFailure
		}
	}
	if bindingToModify == nil {
		if c.readonly {
			bindingToModify = &cloudresourcemanager.Binding{Role: bigtableReadonlyRole}
		} else {
			bindingToModify = &cloudresourcemanager.Binding{Role: bigtableRole}
		}
		policy.Bindings = append(policy.Bindings, bindingToModify)
	}
	bindingToModify.Members = append(bindingToModify.Members, sa)
	err = c.cp.SetProjectPolicy(policy)
	if err != nil {
		log.Fatalf("Error setting the IAM policy for project %q: %v", c.cfg.Project, err)
	}
	return subcommands.ExitSuccess
}

type authAddGcs struct {
	cfg      *config.Config
	cp       AuthResourceManagementCP
	readonly bool
	skipConf bool
}

func (authAddGcs) Name() string {
	return "add-gcs"
}

func (authAddGcs) Synopsis() string {
	return "ensures Cloud TPUs are authorized for Cloud Storage"
}

func (authAddGcs) Usage() string {
	return `ctpu auth add-gcs [BUCKET]

If you supply a bucket name (recommended), permissions are only modified on
the specified bucket instead of authorizing Cloud TPUs to read and write to
all buckets in the project.

`
}

func (c *authAddGcs) SetFlags(f *flag.FlagSet) {
	c.cfg.SetFlags(f)
	f.BoolVar(&c.readonly, "readonly", false, "If modifying per-bucket ACLs, use read-only permissions instead of read-write permissions.")
	f.BoolVar(&c.skipConf, "skip-confirmation", false, "Skip confirmation before changes are made.")
}

func (c *authAddGcs) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	project, err := c.cp.GetProject()
	if err != nil {
		log.Fatalf("Error retrieving project metadata for project %q: %v", c.cfg.Project, err)
	}
	sa := makeTPUServiceName(project)

	policy, err := c.cp.GetProjectPolicy()
	if err != nil {
		log.Fatalf("Error retrieving project IAM policy for project: %q: %v", c.cfg.Project, err)
	}

	var bindingToModify *cloudresourcemanager.Binding
	for _, b := range policy.Bindings {
		if b.Role == storageRole {
			if bindingHasServiceAccount(sa, b) {
				fmt.Printf("Cloud TPUs in project %q already have access to Cloud Storage.\nNo changes were made.\n", c.cfg.Project)
				return subcommands.ExitSuccess
			}
			bindingToModify = b
		}
	}

	if flags.NArg() == 0 {
		if c.readonly {
			fmt.Printf("--readonly is not compatible with project-wide IAM roles.\n")
			return subcommands.ExitUsageError
		}
		if !c.skipConf {
			ok, err := askForConfirmation("Are you sure you want to set project-wide permissions?")
			if err != nil {
				log.Fatalf("Error while asking for confirmation: %v", err)
				return subcommands.ExitFailure
			}
			if !ok {
				fmt.Printf("No changes have been made; exiting!\n")
				return subcommands.ExitUsageError
			}
		}

		// Set project-wide permissions.
		if bindingToModify == nil {
			bindingToModify = &cloudresourcemanager.Binding{Role: storageRole}
			policy.Bindings = append(policy.Bindings, bindingToModify)
		}
		bindingToModify.Members = append(bindingToModify.Members, sa)
		err = c.cp.SetProjectPolicy(policy)
		if err != nil {
			log.Fatalf("Error setting the IAM policy for project %q: %v", c.cfg.Project, err)
		}
		return subcommands.ExitSuccess
	} else if flags.NArg() == 1 {
		bucketName := flags.Arg(0)
		aclRules, err := c.cp.GetBucketACL(ctx, bucketName)
		if err != nil {
			log.Fatalf("Could not retrieve ACLs for bucket %q: %v", bucketName, err)
		}

		tpuEntity := storage.ACLEntity(fmt.Sprintf("user-service-%d@cloud-tpu.iam.gserviceaccount.com", project.ProjectNumber))

		for _, r := range aclRules {
			if r.Entity == tpuEntity {
				if r.Role == storage.RoleOwner || r.Role == roleWriter || (c.readonly && r.Role == storage.RoleReader) {
					fmt.Printf("Cloud TPUs in project %q already have permissions on bucket %q.\nNo changes have been made.\n", c.cfg.Project, bucketName)
					return subcommands.ExitSuccess
				}
			}
		}

		setRole := storage.RoleOwner
		if c.readonly {
			setRole = storage.RoleReader
		}
		err = c.cp.SetBucketACL(ctx, bucketName, tpuEntity, setRole)

		if err != nil {
			log.Fatalf("Error setting ACL on bucket %q: %v", bucketName, err)
		}

		return subcommands.ExitSuccess
	} else {
		fmt.Printf("Usage error: too many arguments supplied.\n")
		return subcommands.ExitUsageError
	}
}

func makeTPUServiceName(p *cloudresourcemanager.Project) string {
	return fmt.Sprintf(serviceAgentFormatString, p.ProjectNumber)
}

func bindingHasServiceAccount(sa string, binding *cloudresourcemanager.Binding) bool {
	for _, member := range binding.Members {
		if member == sa {
			return true
		}
	}
	return false
}

func filterBindings(sa string, policy *cloudresourcemanager.Policy) []*cloudresourcemanager.Binding {
	var matching []*cloudresourcemanager.Binding
	for _, binding := range policy.Bindings {
		if bindingHasServiceAccount(sa, binding) {
			matching = append(matching, binding)
		}
	}

	return matching
}
