"""Tool registry for managing available tools"""

from typing import List, Dict, Any, Optional, Tuple
from .tool_matcher import ToolMatcher


class ToolRegistry:
    """Registry for managing and searching tools"""
    
    def __init__(self, max_tools: int = 3, min_similarity: float = 0.1):
        self.tools = []
        self.categories = {}
        self.matcher = ToolMatcher(max_tools, min_similarity)
        self._needs_refit = False
    
    def register_tool(self, tool: Dict[str, Any], category: str = "general"):
        """Register a new tool"""
        if "function" not in tool:
            return
        
        func = tool["function"]
        if "name" not in func:
            return
        
        # Add category
        tool["category"] = category
        
        # Remove duplicate
        existing = self.get_tool_by_name(func["name"])
        if existing:
            self.tools.remove(existing)
        
        self.tools.append(tool)
        
        # Update categories
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(func["name"])
        
        # Mark that refit is needed (lazy retraining)
        self._needs_refit = True
    
    def register_tools(self, tools: List[Dict[str, Any]], category: str = "general"):
        """Register multiple tools efficiently"""
        for tool in tools:
            # Use internal method to avoid multiple refits
            self._register_tool_internal(tool, category)
        # Single refit after all tools are registered
        self._needs_refit = True
    
    def _register_tool_internal(self, tool: Dict[str, Any], category: str = "general"):
        """Internal method for registering tool without refit"""
        if "function" not in tool:
            return
        
        func = tool["function"]
        if "name" not in func:
            return
        
        # Add category
        tool["category"] = category
        
        # Remove duplicate
        existing = self.get_tool_by_name(func["name"])
        if existing:
            self.tools.remove(existing)
        
        self.tools.append(tool)
        
        # Update categories
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(func["name"])
    
    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool by name"""
        for tool in self.tools:
            func = tool.get("function", {})
            if func.get("name") == name:
                return tool
        return None
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get tools by category"""
        return [tool for tool in self.tools if tool.get("category") == category]
    
    def search_tools(self, query: str, category: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search for tools with lazy refit"""
        # Ensure matcher is up to date
        if self._needs_refit:
            self.matcher.fit(self.tools)
            self._needs_refit = False
        
        if category:
            tools = self.get_tools_by_category(category)
            temp_matcher = ToolMatcher(self.matcher.max_tools, self.matcher.min_similarity)
            temp_matcher.fit(tools)
            return temp_matcher.match_tools(query)
        else:
            return self.matcher.match_tools(query)
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools"""
        return self.tools.copy()
    
    def get_categories(self) -> List[str]:
        """Get all categories"""
        return list(self.categories.keys())
    
    def clear_cache(self):
        """Clear matcher cache for fresh start"""
        if hasattr(self.matcher, '_query_cache'):
            self.matcher._query_cache.clear()
        self._needs_refit = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "categories": len(self.categories),
            "category_breakdown": {cat: len(tools) for cat, tools in self.categories.items()},
            "needs_refit": self._needs_refit,
            "cache_size": len(getattr(self.matcher, '_query_cache', {}))
        }
