# AWS CI infrastructure

This directory contains the source templates for AWS resources used by Newton CI. Keeping the templates in Git provides a reviewable history and a known version to restore during an incident. These files are source files, not examples or generated output.

The templates contain no credentials, secrets, private endpoints, or notification subscribers. A Git commit does not deploy AWS resources or change AWS access. Deploying the watchdog template creates an IAM role with the permissions described below.

## AWS terms used here

AWS has a name for almost every part of this setup. These definitions describe what each term means for the Newton CI watchdog.

| Term | Meaning in this README |
| --- | --- |
| AWS account | The account owns these resources and is charged for what they use. Before an update, `sts get-caller-identity` helps you check that the AWS CLI is pointed at the Newton CI account. |
| [Region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html) | A separate AWS location with a code such as `us-east-1`. Resources in one Region usually do not appear in another. The watchdog is deployed in `us-east-1` but checks runners in several Regions. |
| [AWS CLI profile](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) | A named set of credentials and settings for AWS commands. `AWS_PROFILE` selects the profile, which determines the identity and account used by the commands below. |
| [EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html) and [EBS volume](https://docs.aws.amazon.com/ebs/latest/userguide/what-is-ebs.html) | An EC2 instance is the virtual machine that runs a CI job. An EBS volume is its attached disk. The runner workflow tags both so operators can trace them back to GitHub. |
| [CloudFormation template, stack, and change set](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cloudformation-overview.html) | A template is a YAML file that describes AWS resources. A stack is the live group of resources that CloudFormation manages from the template. A change set previews an update; creating one does not apply the proposed changes. |
| [IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) and [policy](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html) | The role is the identity that Lambda uses when it calls AWS. Its policy says which API actions are allowed and under what conditions. Here, that means scanning EC2 and terminating only instances with the ownership tag. |
| [Lambda](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html) | The service that runs the watchdog's Python code without a dedicated server. Each scheduled check is one Lambda invocation. |
| [EventBridge](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-create-rule-schedule.html) | The timer for the watchdog. Its scheduled rule invokes the Lambda function every 15 minutes. |
| [CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html) | The service that stores the Lambda logs and the numbers reported by the watchdog. CloudWatch alarms watch those metrics and Lambda errors, then send alarm state changes to SNS. |
| [SNS topic](https://docs.aws.amazon.com/sns/latest/dg/welcome.html) and [ARN](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html) | An SNS topic sends alarm notifications to its subscribers. An ARN is the unique AWS identifier for a resource. `AlertTopicArn` points the stack at an existing topic without putting its subscribers in this repository. |
| [Resource tag](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_Tags.html) and ownership boundary | A tag is a key-value label on an AWS resource. The watchdog searches for `created-by=github-actions-newton-role`, and its IAM policy checks the same tag before termination. That repeated check is the ownership boundary. |
| [Control plane](https://docs.aws.amazon.com/whitepapers/latest/aws-fault-isolation-boundaries/control-planes-and-data-planes.html) | The administrative APIs that create, change, describe, and delete AWS resources. The watchdog uses EC2 control-plane calls to find and terminate runners in monitored Regions. Its own stack lives in `us-east-1`. |
| Shadow mode, canary, and rollback | Shadow mode records what the watchdog would terminate without doing it. A canary limits real termination to one repository before wider cleanup is enabled. A rollback restores the saved template and parameters through a reverse change set. |

## Runner attribution tags

The runner workflows apply repository, trigger, workload, and run identifiers as resource tags on EC2 instances and attached EBS volumes. These tags support resource-level attribution through the EC2 console and APIs. The AWS account used by Newton CI does not allow user-defined cost-allocation tags, so these keys cannot be activated as Cost Explorer or Cost and Usage Report dimensions.

## Inventory and safety boundaries

| Template | Stack | Region | Required parameters |
| --- | --- | --- | --- |
| `overdue-newton-github-runner-watchdog.yaml` | `overdue-newton-github-runner-watchdog` | `us-east-1` | `AlertTopicArn` |

`AlertTopicArn` is the only required template parameter. It has no default and must identify the existing SNS topic that receives watchdog alarm transitions. The optional parameters have these defaults:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `TerminateAfterMinutes` | `90` | Age at which an owned running instance becomes a cleanup candidate. Accepts values from `1` through `120`. |
| `TerminationEnabled` | `false` | Enables termination when set to `true`. `false` is shadow mode, which records instances that the watchdog would terminate without terminating them. |
| `RequiredRepository` | Empty string | Requires an exact repository tag when nonempty. It must be set when termination is enabled with `TerminateAfterMinutes` below `90`. It is additional to, not a replacement for, the permanent ownership boundary. |

The permanent ownership boundary is `created-by=github-actions-newton-role`. The watchdog discovers only instances with this tag, and its IAM policy permits termination only for instances with the same tag.

The watchdog stack runs in `us-east-1`. Its EventBridge rule, Lambda function, logs, metrics, and alarms are regional resources, so the stack has one home region. The Lambda function creates boto3 EC2 clients for each monitored region: `us-east-1`, `us-east-2`, `us-west-2`, `ap-northeast-1`, and `ap-northeast-2`.

## Validate and review a change

Run [`cfn-lint`](https://github.com/aws-cloudformation/cfn-lint) locally before signing in to AWS. It checks resource types and properties against the CloudFormation schemas:

```bash
uvx --from cfn-lint cfn-lint \
  scripts/ci/aws/overdue-newton-github-runner-watchdog.yaml
```

Next, authenticate the AWS CLI and confirm that the selected profile is for the intended account. `validate-template` sends the template to CloudFormation for a syntax check. It does not replace the schema checks from `cfn-lint`:

```bash
if [ -z "${AWS_PROFILE:-}" ]; then
  echo "AWS_PROFILE must name the intended AWS CLI profile" >&2
  exit 1
fi

AWS_REGION=us-east-1
WATCHDOG_STACK=overdue-newton-github-runner-watchdog
CHANGE_SET=runner-watchdog-update
EVIDENCE_DIR="$(mktemp -d)" || exit 1

aws --profile "$AWS_PROFILE" sts get-caller-identity
aws --profile "$AWS_PROFILE" cloudformation validate-template \
  --region "$AWS_REGION" \
  --template-body file://scripts/ci/aws/overdue-newton-github-runner-watchdog.yaml
```

Neither command tests the Python code embedded in the Lambda resource. `scripts/ci/tests/test_watchdog_logic.py` covers that behavior.

Save the current template, parameters, and stack status outside the repository before an update. Keep that evidence until the change is no longer needed for rollback.

```bash
aws --profile "$AWS_PROFILE" cloudformation get-template \
  --region "$AWS_REGION" \
  --stack-name "$WATCHDOG_STACK" \
  --query TemplateBody \
  --output text > "$EVIDENCE_DIR/current-template.yaml" || exit 1

[ -s "$EVIDENCE_DIR/current-template.yaml" ] || {
  echo "Saved template is empty" >&2
  exit 1
}

aws --profile "$AWS_PROFILE" cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$WATCHDOG_STACK" \
  --query 'Stacks[0].{Parameters:Parameters,StackStatus:StackStatus}' \
  --output json > "$EVIDENCE_DIR/current-stack.json" || exit 1

[ -s "$EVIDENCE_DIR/current-stack.json" ] || {
  echo "Saved stack configuration is empty" >&2
  exit 1
}
```

Use the committed template and a reviewed CloudFormation change set. This update example keeps cleanup disabled explicitly. `AlertTopicArn` uses the current stack value; creating a new stack requires supplying that required parameter with the real SNS topic ARN outside this document.

```bash
aws --profile "$AWS_PROFILE" cloudformation create-change-set \
  --region "$AWS_REGION" \
  --stack-name "$WATCHDOG_STACK" \
  --change-set-name "$CHANGE_SET" \
  --change-set-type UPDATE \
  --template-body file://scripts/ci/aws/overdue-newton-github-runner-watchdog.yaml \
  --parameters \
    'ParameterKey=AlertTopicArn,UsePreviousValue=true' \
    'ParameterKey=TerminateAfterMinutes,ParameterValue=90' \
    'ParameterKey=TerminationEnabled,ParameterValue=false' \
    'ParameterKey=RequiredRepository,ParameterValue=' \
  --capabilities CAPABILITY_NAMED_IAM

aws --profile "$AWS_PROFILE" cloudformation wait change-set-create-complete \
  --region "$AWS_REGION" \
  --stack-name "$WATCHDOG_STACK" \
  --change-set-name "$CHANGE_SET"

aws --profile "$AWS_PROFILE" cloudformation describe-change-set \
  --region "$AWS_REGION" \
  --stack-name "$WATCHDOG_STACK" \
  --change-set-name "$CHANGE_SET"
```

The stack has a named IAM role, so every change set requires `CAPABILITY_NAMED_IAM`. Review the full change set before execution and stop if it includes an unexpected replacement, deletion, resource, or permission change. Operators enable cleanup only through reviewed parameter changes.

```bash
aws --profile "$AWS_PROFILE" cloudformation execute-change-set \
  --region "$AWS_REGION" \
  --stack-name "$WATCHDOG_STACK" \
  --change-set-name "$CHANGE_SET"

aws --profile "$AWS_PROFILE" cloudformation wait stack-update-complete \
  --region "$AWS_REGION" \
  --stack-name "$WATCHDOG_STACK"
```

Local tests can exercise template behavior, but they cannot prove AWS IAM enforcement.

## Failure handling and rollback

The watchdog collects regional scan and per-instance termination failures so it can continue the remaining work. Any collected failure causes the invocation to fail after the watchdog emits metrics and structured output. The native Lambda error alarm catches those failed invocations.

To roll back a deployed stack change, create and review a reverse CloudFormation change set using the saved template and parameters, then execute it. One-time migration rollback and legacy-resource handling are intentionally outside this permanent README.
