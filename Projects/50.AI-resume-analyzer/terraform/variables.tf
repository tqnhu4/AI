# variables.tf
variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "ap-southeast-1" # Change to your desired AWS region
}

variable "project_name" {
  description = "Overall project name"
  type        = string
  default     = "ai-resume-analyzer"
}

variable "environment_name" {
  description = "Environment name (e.g., dev, prod)"
  type        = string
  default     = "dev"
}

variable "gemini_api_key" {
  description = "Gemini API key to access generative AI models"
  type        = string
  sensitive   = true # Mark as sensitive so Terraform doesn't display it in output
}