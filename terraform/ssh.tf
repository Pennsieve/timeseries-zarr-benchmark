resource "tls_private_key" "operator" {
  algorithm = "ED25519"
}

resource "aws_key_pair" "operator" {
  key_name   = var.name_prefix
  public_key = tls_private_key.operator.public_key_openssh
}
