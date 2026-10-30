variable "region" {
  description = "AWS region of every resource, the bucket included."
  type        = string
  default     = "us-east-1"
}

variable "availability_zone" {
  description = "Availability zone of every instance and volume. The first zone of the region when null."
  type        = string
  default     = null
}

variable "name_prefix" {
  description = "Prefix of every resource name and the value of the Project tag."
  type        = string
  default     = "tszbench"
}

variable "operator_cidr" {
  description = "CIDR block allowed to open SSH connections, for example 203.0.113.7/32."
  type        = string

  validation {
    condition     = can(cidrhost(var.operator_cidr, 0))
    error_message = "operator_cidr must be a CIDR block such as 203.0.113.7/32."
  }
}

variable "vpc_cidr" {
  description = "CIDR block of the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR block of the one public subnet. It must lie inside vpc_cidr."
  type        = string
  default     = "10.20.1.0/24"
}

variable "bucket_name" {
  description = "Name of the S3 bucket. A name made of name_prefix, the account id, and the region when null."
  type        = string
  default     = null
}

variable "bucket_force_destroy" {
  description = "Delete the bucket with its objects on destroy."
  type        = bool
  default     = false
}

variable "ami_id" {
  description = "AMI of every instance. The newest Ubuntu 24.04 amd64 image of Canonical when null. Set it to repeat a run on the same image."
  type        = string
  default     = null
}

variable "login_user" {
  description = "Login user of the AMI. It owns the repository clone and may run drop-caches."
  type        = string
  default     = "ubuntu"
}

variable "root_volume_gb" {
  description = "Size of the root volume of every instance in GiB."
  type        = number
  default     = 50
}

variable "file_server_instance_type" {
  description = "Instance type of the file server. The c6i.8xlarge has a fixed 12.5 Gbps network and 10 Gbps EBS bandwidth. Smaller c6i types get full bandwidth only in bursts."
  type        = string
  default     = "c6i.8xlarge"
}

variable "client_instance_type" {
  description = "Instance type of the benchmark client. It has 32 vCPU, which the generate stage uses as workers. Smaller c6i types get full bandwidth only in bursts."
  type        = string
  default     = "c6i.8xlarge"
}

variable "resampling_server_instance_type" {
  description = "Instance type of the resampling server."
  type        = string
  default     = "c6i.8xlarge"
}

variable "load_generator_instance_type" {
  description = "Instance type of each load generator."
  type        = string
  default     = "c6i.8xlarge"
}

variable "load_generator_count" {
  description = "Number of load generators. A 1000-user run uses 10 generators of 100 users, and its repeat with half the users per generator uses 20 of 50. Set 0 to create the other machines first."
  type        = number
  default     = 20

  validation {
    condition     = var.load_generator_count >= 0 && var.load_generator_count <= 20 && floor(var.load_generator_count) == var.load_generator_count
    error_message = "load_generator_count must be a whole number from 0 to 20."
  }
}

variable "file_server_data_gb" {
  description = "Size of the data volume of the file server in GiB. It holds every format of every recording."
  type        = number
  default     = 2000
}

variable "client_data_gb" {
  description = "Size of the data volume of the benchmark client in GiB. It holds the local copy."
  type        = number
  default     = 2000
}

variable "data_volume_iops" {
  description = "Provisioned IOPS of each data volume. 16000 is the gp3 maximum."
  type        = number
  default     = 16000

  validation {
    condition     = var.data_volume_iops >= 3000 && var.data_volume_iops <= 16000
    error_message = "data_volume_iops must be from 3000 to 16000, the gp3 range."
  }
}

variable "data_volume_throughput_mibps" {
  description = "Provisioned throughput of each data volume in MiB/s. 1000 is the gp3 maximum."
  type        = number
  default     = 1000

  validation {
    condition     = var.data_volume_throughput_mibps >= 125 && var.data_volume_throughput_mibps <= 1000
    error_message = "data_volume_throughput_mibps must be from 125 to 1000, the gp3 range."
  }
}

variable "repo_url" {
  description = "Git URL of the benchmark repository that every instance clones."
  type        = string
}

variable "repo_commit" {
  description = "Commit, tag, or branch of the benchmark repository to check out."
  type        = string
}
