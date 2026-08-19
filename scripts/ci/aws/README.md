# AWS CI infrastructure

This directory contains the source templates for AWS resources used by Newton
CI. If the templates exist only in AWS, maintainers have to reconstruct the
deployed configuration from console history during an incident. Checking them
into Git gives us a reviewable change history and a known version to restore.

That history is why these files belong in the repository. They are source files,
not examples or generated output. The templates contain no credentials,
secrets, private endpoints, or notification subscribers. They do not grant
access to AWS, and commits to this directory do not deploy automatically.

Put future AWS CI templates here too. This gives the small group that maintains
the infrastructure one place to look and keeps AWS details out of the rest of
the project.

## Runner attribution tags

The runner workflows apply repository, trigger, workload, and run identifiers
as resource tags on EC2 instances and attached EBS volumes. These tags support
resource-level attribution through the EC2 console and APIs. The AWS account
used by Newton CI does not allow user-defined cost-allocation tags, so these
keys cannot be activated as Cost Explorer or Cost and Usage Report dimensions.

## Inventory

| Template | Stack | Region | Required parameters |
| --- | --- | --- | --- |
| `overdue-newton-github-runner-watchdog.yaml` | `overdue-newton-github-runner-watchdog` | `us-east-1` | `AlertTopicArn` |

The deployer must supply `AlertTopicArn` because the template has no default.
This keeps notification routing and subscriber details outside the repository.

## Validate and review a change

Authenticate the AWS CLI, check that the selected profile points to the intended
account, then validate the template:

```bash
NEWTON_AWS_PROFILE=isaac-sim-CS-Admin
NEWTON_AWS_REGION=us-east-1
NEWTON_WATCHDOG_STACK=overdue-newton-github-runner-watchdog
NEWTON_CHANGE_SET=aws-runner-watchdog-update

aws --profile "$NEWTON_AWS_PROFILE" sts get-caller-identity
aws --profile "$NEWTON_AWS_PROFILE" cloudformation validate-template \
  --region "$NEWTON_AWS_REGION" \
  --template-body file://scripts/ci/aws/overdue-newton-github-runner-watchdog.yaml
```

Before creating a change set, save the stack's current template and parameters
outside the repository in case you need to roll back. Read the alert action on
the deployed alarm and verify that it points to the intended SNS topic:

```bash
NEWTON_ALERT_TOPIC_ARN="$(
  aws --profile "$NEWTON_AWS_PROFILE" cloudwatch describe-alarms \
    --region "$NEWTON_AWS_REGION" \
    --alarm-names overdue-newton-github-runner-watchdog \
    --query 'MetricAlarms[0].AlarmActions[0]' \
    --output text
)"

case "$NEWTON_ALERT_TOPIC_ARN" in
  arn:aws:sns:us-east-1:*) ;;
  *) echo "Unexpected alert topic ARN" >&2; exit 1 ;;
esac

aws --profile "$NEWTON_AWS_PROFILE" cloudformation create-change-set \
  --region "$NEWTON_AWS_REGION" \
  --stack-name "$NEWTON_WATCHDOG_STACK" \
  --change-set-name "$NEWTON_CHANGE_SET" \
  --change-set-type UPDATE \
  --template-body file://scripts/ci/aws/overdue-newton-github-runner-watchdog.yaml \
  --parameters "ParameterKey=AlertTopicArn,ParameterValue=$NEWTON_ALERT_TOPIC_ARN" \
  --capabilities CAPABILITY_NAMED_IAM

aws --profile "$NEWTON_AWS_PROFILE" cloudformation wait change-set-create-complete \
  --region "$NEWTON_AWS_REGION" \
  --stack-name "$NEWTON_WATCHDOG_STACK" \
  --change-set-name "$NEWTON_CHANGE_SET"

aws --profile "$NEWTON_AWS_PROFILE" cloudformation describe-change-set \
  --region "$NEWTON_AWS_REGION" \
  --stack-name "$NEWTON_WATCHDOG_STACK" \
  --change-set-name "$NEWTON_CHANGE_SET"
```

Read the complete change set before executing it. This stack has a named IAM
role, so the update requires `CAPABILITY_NAMED_IAM`. Pay close attention to IAM
changes, and stop if the change set contains an unexpected replacement,
deletion, resource, or permission change.

Execute only after review:

```bash
aws --profile "$NEWTON_AWS_PROFILE" cloudformation execute-change-set \
  --region "$NEWTON_AWS_REGION" \
  --stack-name "$NEWTON_WATCHDOG_STACK" \
  --change-set-name "$NEWTON_CHANGE_SET"

aws --profile "$NEWTON_AWS_PROFILE" cloudformation wait stack-update-complete \
  --region "$NEWTON_AWS_REGION" \
  --stack-name "$NEWTON_WATCHDOG_STACK"
```

After deployment, invoke the watchdog, confirm that it emits fresh metrics and
logs, then check the alarm state. Investigate any stack drift before making
another change. If verification fails, restore the saved template and parameters
with a reviewed reverse change set. Do not update stack-managed resources
directly because CloudFormation will no longer have an accurate record of their
configuration.
