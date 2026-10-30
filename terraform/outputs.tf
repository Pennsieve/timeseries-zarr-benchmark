locals {
  load_generators = aws_instance.load_generator[*]
}

output "region" {
  description = "Region of every resource."
  value       = var.region
}

output "availability_zone" {
  description = "Availability zone of every instance."
  value       = local.availability_zone
}

output "ami_id" {
  description = "AMI of every instance. Set it as ami_id in terraform.tfvars to keep later applies on the same image."
  value       = local.ami_id
}

output "bucket_name" {
  description = "Name of the S3 bucket."
  value       = aws_s3_bucket.data.bucket
}

output "ssh_user" {
  description = "Login user of every instance."
  value       = var.login_user
}

output "public_ips" {
  description = "Public IP address of each instance, by role."
  value = {
    file_server       = aws_instance.file_server.public_ip
    client            = aws_instance.client.public_ip
    resampling_server = aws_instance.resampling_server.public_ip
    load_generators   = local.load_generators[*].public_ip
  }
}

output "private_ips" {
  description = "Private IP address of each instance, by role."
  value = {
    file_server       = aws_instance.file_server.private_ip
    client            = aws_instance.client.private_ip
    resampling_server = aws_instance.resampling_server.private_ip
    load_generators   = local.load_generators[*].private_ip
  }
}

output "instance_ids" {
  description = "Instance id of each instance, by role."
  value = {
    file_server       = aws_instance.file_server.id
    client            = aws_instance.client.id
    resampling_server = aws_instance.resampling_server.id
    load_generators   = local.load_generators[*].id
  }
}

output "hosts_file" {
  description = "One line per instance for the orchestrator: role, a space, and the private IP address. The roles are file_server, client, resampling_server, and load_generator."
  value = join("\n", concat(
    [
      "file_server ${aws_instance.file_server.private_ip}",
      "client ${aws_instance.client.private_ip}",
      "resampling_server ${aws_instance.resampling_server.private_ip}",
    ],
    [for instance in local.load_generators : "load_generator ${instance.private_ip}"],
  ))
}

output "nginx_version_path" {
  description = "Path of the nginx version and build file on the file server, and its URL path under the HTTP root."
  value = {
    local_path = "/data/nginx-version.txt"
    url_path   = "/nginx-version.txt"
  }
}

output "ssh_private_key" {
  description = "OpenSSH private key of every instance. Save it with mode 0600 and copy it to the client."
  value       = tls_private_key.operator.private_key_openssh
  sensitive   = true
}
