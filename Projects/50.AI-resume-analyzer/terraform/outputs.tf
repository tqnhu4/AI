# outputs.tf
output "api_gateway_invoke_url" {
  description = "URL to invoke the API Gateway"
  value       = aws_apigatewayv2_stage.dev_stage.invoke_url
}

output "lambda_function_name" {
  description = "Name of the deployed Lambda function"
  value       = aws_lambda_function.resume_analyzer_lambda.function_name
}

output "s3_code_bucket_name" {
  description = "Name of the S3 bucket containing Lambda source code"
  value       = aws_s3_bucket.lambda_code_bucket.id
}