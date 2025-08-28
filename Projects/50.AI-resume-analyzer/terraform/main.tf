# main.tf
# Terraform Configuration for AI Resume Analyzer Backend

# 1. AWS Provider Configuration
provider "aws" {
  region = var.aws_region
}

# 2. S3 Bucket to store Lambda source code
resource "aws_s3_bucket" "lambda_code_bucket" {
  bucket = "${var.project_name}-${var.environment_name}-lambda-code-${data.aws_caller_identity.current.account_id}"
  acl    = "private"

  versioning {
    enabled = true
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment_name
    Service     = "LambdaCode"
  }
}

# 3. IAM Role for Lambda Function
resource "aws_iam_role" "lambda_exec_role" {
  name = "${var.project_name}-${var.environment_name}-LambdaExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment_name
  }
}

# 4. IAM Policy for Lambda Function (logs and S3 access)
resource "aws_iam_policy" "lambda_policy" {
  name        = "${var.project_name}-${var.environment_name}-LambdaPolicy"
  description = "IAM Policy for AI Resume Analyzer Lambda function"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/*:*"
      },
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject" # If Lambda needs to write to S3 (e.g., model cache)
        ],
        Resource = "${aws_s3_bucket.lambda_code_bucket.arn}/*"
      }
      # Add other permissions if Lambda needs to interact with other AWS services
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy_attach" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}

# Data source to get current Account ID
data "aws_caller_identity" "current" {}

# 5. Lambda Function
resource "aws_lambda_function" "resume_analyzer_lambda" {
  function_name = "${var.project_name}-${var.environment_name}-ResumeAnalyzer"
  handler       = "wsgi_handler.lambda_handler" # File name.handler name (points to WSGI handler)
  runtime       = "python3.9" # Or python3.10/3.11/3.12
  role          = aws_iam_role.lambda_exec_role.arn
  timeout       = 60 # Increase timeout as AI processing can take time
  memory_size   = 512 # Increase memory for better NLP/Embedding performance

  s3_bucket = aws_s3_bucket.lambda_code_bucket.id
  s3_key    = "lambda_package.zip" # Name of the zip file containing code and dependencies

  environment {
    variables = {
      GEMINI_API_KEY = var.gemini_api_key # Pass API Key to Lambda
      # Other environment variables if needed
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment_name
    Service     = "ResumeAnalyzer"
  }
}

# 6. API Gateway HTTP API
resource "aws_apigatewayv2_api" "http_api" {
  name          = "${var.project_name}-${var.environment_name}-HttpApi"
  protocol_type = "HTTP"

  tags = {
    Project     = var.project_name
    Environment = var.environment_name
  }
}

# 7. API Gateway Integration (HTTP API -> Lambda)
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id             = aws_apigatewayv2_api.http_api.id
  integration_type   = "AWS_PROXY"
  integration_method = "POST"
  integration_uri    = aws_lambda_function.resume_analyzer_lambda.invoke_arn
}

# 8. API Gateway Route: /analyze-cv (POST)
resource "aws_apigatewayv2_route" "analyze_cv_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /analyze-cv"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# 9. API Gateway Route: /interview (POST)
resource "aws_apigatewayv2_route" "interview_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /interview"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# 10. API Gateway Stage (dev)
resource "aws_apigatewayv2_stage" "dev_stage" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = var.environment_name
  auto_deploy = true # Automatically deploy on changes
}

# 11. Permission for API Gateway to invoke Lambda
resource "aws_lambda_permission" "api_gateway_permission" {
  statement_id  = "AllowAPIGatewayInvokeLambda"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.resume_analyzer_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  # source_arn to restrict invocation only from your API Gateway
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}