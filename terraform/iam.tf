data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "bucket_read" {
  statement {
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.data.arn]
  }

  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
}

data "aws_iam_policy_document" "bucket_write" {
  statement {
    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:AbortMultipartUpload",
    ]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }

  statement {
    actions   = ["s3:ListBucketMultipartUploads"]
    resources = [aws_s3_bucket.data.arn]
  }
}

resource "aws_iam_role" "reader" {
  name               = "${var.name_prefix}-reader"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

resource "aws_iam_role_policy" "reader_bucket" {
  name   = "bucket-read"
  role   = aws_iam_role.reader.id
  policy = data.aws_iam_policy_document.bucket_read.json
}

resource "aws_iam_instance_profile" "reader" {
  name = "${var.name_prefix}-reader"
  role = aws_iam_role.reader.name
}

resource "aws_iam_role" "writer" {
  name               = "${var.name_prefix}-writer"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

resource "aws_iam_role_policy" "writer_bucket_read" {
  name   = "bucket-read"
  role   = aws_iam_role.writer.id
  policy = data.aws_iam_policy_document.bucket_read.json
}

resource "aws_iam_role_policy" "writer_bucket_write" {
  name   = "bucket-write"
  role   = aws_iam_role.writer.id
  policy = data.aws_iam_policy_document.bucket_write.json
}

resource "aws_iam_instance_profile" "writer" {
  name = "${var.name_prefix}-writer"
  role = aws_iam_role.writer.name
}
