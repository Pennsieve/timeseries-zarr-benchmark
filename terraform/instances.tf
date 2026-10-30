data "aws_ami" "ubuntu" {
  count       = var.ami_id == null ? 1 : 0
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  ami_id = var.ami_id != null ? var.ami_id : data.aws_ami.ubuntu[0].id

  script_vars = {
    login_user  = var.login_user
    bucket      = aws_s3_bucket.data.bucket
    region      = var.region
    repo_url    = var.repo_url
    repo_commit = var.repo_commit
  }

  repository_script = templatefile("${path.module}/user_data/repository.sh.tftpl", local.script_vars)

  data_volume_script = templatefile("${path.module}/user_data/data_volume.sh.tftpl", local.script_vars)

  nginx_script = templatefile(
    "${path.module}/user_data/nginx.sh.tftpl",
    merge(local.script_vars, { nginx_conf = file("${path.module}/user_data/nginx.conf") }),
  )

  packages_script = {
    for role in ["file_server", "client", "resampling_server", "load_generator"] :
    role => templatefile("${path.module}/user_data/packages.sh.tftpl", merge(local.script_vars, { role = role }))
  }

  user_data = {
    file_server       = join("\n", [local.packages_script["file_server"], local.data_volume_script, local.nginx_script, local.repository_script])
    client            = join("\n", [local.packages_script["client"], local.data_volume_script, local.repository_script])
    resampling_server = join("\n", [local.packages_script["resampling_server"], local.repository_script])
    load_generator    = join("\n", [local.packages_script["load_generator"], local.repository_script])
  }
}

resource "aws_instance" "file_server" {
  ami                         = local.ami_id
  instance_type               = var.file_server_instance_type
  subnet_id                   = aws_subnet.public.id
  availability_zone           = local.availability_zone
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  key_name                    = aws_key_pair.operator.key_name
  iam_instance_profile        = aws_iam_instance_profile.reader.name
  associate_public_ip_address = true
  user_data                   = local.user_data["file_server"]
  user_data_replace_on_change = true

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
  }

  ebs_block_device {
    device_name = "/dev/sdf"
    volume_type = "gp3"
    volume_size = var.file_server_data_gb
    iops        = var.data_volume_iops
    throughput  = var.data_volume_throughput_mibps
  }

  tags = {
    Name = "${var.name_prefix}-file-server"
    Role = "file_server"
  }

  # A later apply, such as the one that adds the load generators, must not
  # replace a machine that holds generated data or a running service.
  lifecycle {
    ignore_changes = [ami, user_data]
  }
}

resource "aws_instance" "client" {
  ami                         = local.ami_id
  instance_type               = var.client_instance_type
  subnet_id                   = aws_subnet.public.id
  availability_zone           = local.availability_zone
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  key_name                    = aws_key_pair.operator.key_name
  iam_instance_profile        = aws_iam_instance_profile.writer.name
  associate_public_ip_address = true
  user_data                   = local.user_data["client"]
  user_data_replace_on_change = true

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
  }

  ebs_block_device {
    device_name = "/dev/sdf"
    volume_type = "gp3"
    volume_size = var.client_data_gb
    iops        = var.data_volume_iops
    throughput  = var.data_volume_throughput_mibps
  }

  tags = {
    Name = "${var.name_prefix}-client"
    Role = "client"
  }

  # A later apply, such as the one that adds the load generators, must not
  # replace a machine that holds generated data or a running service.
  lifecycle {
    ignore_changes = [ami, user_data]
  }
}

resource "aws_instance" "resampling_server" {
  ami                         = local.ami_id
  instance_type               = var.resampling_server_instance_type
  subnet_id                   = aws_subnet.public.id
  availability_zone           = local.availability_zone
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  key_name                    = aws_key_pair.operator.key_name
  iam_instance_profile        = aws_iam_instance_profile.reader.name
  associate_public_ip_address = true
  user_data                   = local.user_data["resampling_server"]
  user_data_replace_on_change = true

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
  }

  tags = {
    Name = "${var.name_prefix}-resampling-server"
    Role = "resampling_server"
  }

  # A later apply, such as the one that adds the load generators, must not
  # replace a machine that holds generated data or a running service.
  lifecycle {
    ignore_changes = [ami, user_data]
  }
}

resource "aws_instance" "load_generator" {
  count                       = var.load_generator_count
  ami                         = local.ami_id
  instance_type               = var.load_generator_instance_type
  subnet_id                   = aws_subnet.public.id
  availability_zone           = local.availability_zone
  vpc_security_group_ids      = [aws_security_group.benchmark.id]
  key_name                    = aws_key_pair.operator.key_name
  iam_instance_profile        = aws_iam_instance_profile.reader.name
  associate_public_ip_address = true
  user_data                   = local.user_data["load_generator"]
  user_data_replace_on_change = true

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
  }

  tags = {
    Name = "${var.name_prefix}-load-generator-${count.index}"
    Role = "load_generator"
  }
}
