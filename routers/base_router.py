#!/usr/bin/env python3
"""
Minimal MCP Base Router implementation for the aggregator server.
This provides a basic JSON-RPC 2.0 handler for MCP methods.
"""

import json
from typing import Any, Dict, Optional, Callable
from fastapi import HTTPException
from fastapi.routing import APIRouter


class MCPBaseRouter:
    """
    Minimal MCP (Model Context Protocol) base router.
    Handles JSON-RPC 2.0 requests and routes MCP methods.
    """
    
    def __init__(self):
        self.router = APIRouter()
        self.methods: Dict[str, Callable] = {}
        
        # Register the JSON-RPC handler
        self.router.add_api_route(
            "/", 
            self._handle_json_rpc, 
            methods=["POST"], 
            response_model=Dict[str, Any]
        )
    
    def register_method(self, method_name: str, handler: Callable):
        """
        Register an MCP method handler.
        
        Args:
            method_name: The name of the MCP method (e.g., "tools/list")
            handler: A callable that takes parameters and returns a result
        """
        self.methods[method_name] = handler
    
    def _handle_json_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle JSON-RPC 2.0 requests.
        
        Args:
            request: JSON-RPC request object
            
        Returns:
            JSON-RPC response object
        """
        try:
            # Validate JSON-RPC request
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            
            if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
                raise ValueError("Invalid JSON-RPC version")
            
            if "method" not in request:
                raise ValueError("Method is required")
            
            if "id" not in request:
                raise ValueError("ID is required")
            
            method = request["method"]
            params = request.get("params", {})
            request_id = request["id"]
            
            # Find and call the method handler
            if method not in self.methods:
                raise ValueError(f"Method '{method}' not found")
            
            handler = self.methods[method]
            result = handler(params)
            
            # Return successful response
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except ValueError as e:
            # Return error response
            return {
                "jsonrpc": "2.0",
                "id": request.get("id", None),
                "error": {
                    "code": -32601,  # Method not found or invalid params
                    "message": str(e)
                }
            }
        except Exception as e:
            # Return error response for unexpected errors
            return {
                "jsonrpc": "2.0",
                "id": request.get("id", None),
                "error": {
                    "code": -32603,  # Internal error
                    "message": f"Internal error: {str(e)}"
                }
            }