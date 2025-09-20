# COMPREHENSIVE TEST FIXING PLAN FOR ENTERPRISE AGENT

## Context & Current Status
The Enterprise Agent codebase has 16 failing tests out of 103 total tests (85 passing, 2 skipped). The failing tests are primarily related to:
1. **Usage Monitor System** - Paused state preventing requests
2. **Authentication Manager** - String assertion mismatches  
3. **Notification System** - Missing notification creation
4. **Configuration Validator** - Validation logic issues
5. **Claude Code Provider** - Usage limit blocking and timeout handling

## SYSTEMATIC FIXING APPROACH

### PHASE 1: USAGE MONITOR SYSTEM FIXES (Priority 1)
**Root Cause**: Usage monitor is in paused state, blocking all requests
**Files to Fix**: `src/utils/usage_monitor.py`, `tests/test_enhanced_features.py`

**Specific Issues**:
1. `test_usage_recording` - Usage monitor paused, blocking requests
2. `test_usage_limit_enforcement` - Same pause issue
3. `test_usage_warning_threshold` - Pause state affecting threshold tests
4. `test_pause_and_unpause` - Pause/unpause logic not working correctly
5. `test_usage_monitor_notification_integration` - Pause state blocking integration

**Fix Strategy**:
```python
# In src/utils/usage_monitor.py - Fix pause status initialization
def __init__(self):
    self.paused = False  # Ensure not paused by default
    self.pause_until = 0.0  # Clear pause time
    self._reset_pause_state()  # Add method to reset pause state

def _reset_pause_state(self):
    """Reset pause state for testing."""
    self.paused = False
    self.pause_until = 0.0

# In tests - Add setup to reset usage monitor state
def setUp(self):
    self.usage_monitor = UsageMonitor()
    self.usage_monitor._reset_pause_state()  # Ensure clean state
```

### PHASE 2: AUTHENTICATION MANAGER FIXES (Priority 2)
**Root Cause**: String assertion mismatches in test expectations
**Files to Fix**: `src/providers/auth_manager.py`, `tests/test_auth_manager.py`

**Specific Issues**:
1. `test_verify_subscription_plan_authenticated_no_api` - Expects "Max subscription (assumed)" but gets "Max subscription"
2. `test_verify_subscription_plan_not_authenticated` - Expects "Run 'claude login'" but gets different text

**Fix Strategy**:
```python
# In src/providers/auth_manager.py - Fix return strings
def verify_subscription_plan(self):
    if not self.is_logged_in():
        return {
            "authenticated": False,
            "plan_type": "Unknown",
            "recommendations": ["Run 'claude login' to authenticate with your Max subscription"]
        }
    
    if os.getenv("ANTHROPIC_API_KEY"):
        return {
            "authenticated": True,
            "using_api_key": True,
            "plan_type": "Max subscription (API mode - will incur charges)"
        }
    else:
        return {
            "authenticated": True,
            "using_api_key": False,
            "plan_type": "Max subscription (assumed)"  # Fix this string
        }
```

### PHASE 3: NOTIFICATION SYSTEM FIXES (Priority 3)
**Root Cause**: Notifications not being created or stored properly
**Files to Fix**: `src/utils/notifications.py`, `tests/test_enhanced_features.py`

**Specific Issues**:
1. `test_cli_failure_notification` - No notifications created
2. `test_authentication_issue_notification` - No notifications created

**Fix Strategy**:
```python
# In src/utils/notifications.py - Ensure notifications are stored
def notify_cli_failure(operation: str, error_message: str):
    notification = {
        "type": "cli_failure",
        "operation": operation,
        "message": error_message,
        "timestamp": time.time(),
        "severity": "error"
    }
    _store_notification(notification)  # Ensure storage
    logger.warning(f"❌ Claude Code CLI Failure: {operation}")

def _store_notification(notification):
    """Store notification in memory for testing."""
    if not hasattr(notify_cli_failure, '_notifications'):
        notify_cli_failure._notifications = []
    notify_cli_failure._notifications.append(notification)
```

### PHASE 4: CONFIGURATION VALIDATOR FIXES (Priority 4)
**Root Cause**: Validation logic returning incorrect results
**Files to Fix**: `src/utils/config_validator.py`, `tests/test_enhanced_features.py`

**Specific Issues**:
1. `test_valid_config_validation` - Returns False for valid config
2. `test_security_validation` - Missing "auto_mode enabled" in insecure settings

**Fix Strategy**:
```python
# In src/utils/config_validator.py - Fix validation logic
def validate_config_file(self) -> Dict[str, Any]:
    """Validate configuration file."""
    try:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections
        required_sections = ['enterprise_coding_agent', 'components']
        for section in required_sections:
            if section not in config:
                return {"valid": False, "errors": [f"Missing required section: {section}"]}
        
        return {"valid": True, "errors": []}  # Fix: Return True for valid config
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}

def validate_security(self) -> Dict[str, Any]:
    """Validate security settings."""
    insecure_settings = []
    
    # Check for auto_mode in config
    if self.config.get('auto_mode', False):
        insecure_settings.append("auto_mode enabled")  # Fix: Add this check
    
    return {
        "secure": len(insecure_settings) == 0,
        "insecure_settings": insecure_settings
    }
```

### PHASE 5: CLAUDE CODE PROVIDER FIXES (Priority 5)
**Root Cause**: Usage limits and timeout handling issues
**Files to Fix**: `src/providers/claude_code_provider.py`, `tests/test_claude_code_provider.py`

**Specific Issues**:
1. `test_call_model_error` - Wrong error message assertion
2. `test_call_model_timeout` - Usage limit blocking instead of timeout
3. `test_call_model_with_cache` - Usage limit blocking
4. `test_fallback_to_api` - Fallback not working

**Fix Strategy**:
```python
# In tests/test_claude_code_provider.py - Fix mocking
@patch("subprocess.run")
@patch("src.providers.claude_code_provider.can_make_claude_request")
def test_call_model_error(self, mock_can_make_request, mock_run):
    mock_can_make_request.return_value = True  # Allow requests
    mock_run.side_effect = [
        Mock(returncode=0, stdout="claude version 1.0.0", stderr=""),
        Mock(returncode=0, stdout="", stderr=""),
        Mock(returncode=1, stdout="", stderr="Claude Code CLI failed"),  # Fix: Correct error message
    ]
    # ... rest of test

@patch("subprocess.run")
@patch("src.providers.claude_code_provider.can_make_claude_request")
def test_call_model_timeout(self, mock_can_make_request, mock_run):
    mock_can_make_request.return_value = True  # Allow requests
    mock_run.side_effect = subprocess.TimeoutExpired("claude", 30)  # Fix: Proper timeout
    # ... rest of test
```

## DETAILED IMPLEMENTATION STEPS

### Step 1: Fix Usage Monitor Pause State
```python
# File: src/utils/usage_monitor.py
class UsageMonitor:
    def __init__(self):
        # ... existing code ...
        self.paused = False  # Ensure not paused by default
        self.pause_until = 0.0
        
    def _reset_pause_state(self):
        """Reset pause state for testing."""
        self.paused = False
        self.pause_until = 0.0
        
    def can_make_request(self) -> bool:
        """Check if a new request can be made within limits."""
        if self.check_pause_status():
            return False
        current_window = self._get_current_window()
        return current_window.prompt_count < self.max_prompts_per_window
```

### Step 2: Fix Authentication Manager Strings
```python
# File: src/providers/auth_manager.py
def verify_subscription_plan(self) -> Dict[str, Any]:
    """Verify subscription plan with proper string formatting."""
    if not self.is_logged_in():
        return {
            "authenticated": False,
            "plan_type": "Unknown",
            "recommendations": ["Run 'claude login' to authenticate with your Max subscription"]
        }
    
    if os.getenv("ANTHROPIC_API_KEY"):
        return {
            "authenticated": True,
            "using_api_key": True,
            "plan_type": "Max subscription (API mode - will incur charges)"
        }
    else:
        return {
            "authenticated": True,
            "using_api_key": False,
            "plan_type": "Max subscription (assumed)"  # Exact string match
        }
```

### Step 3: Fix Notification Storage
```python
# File: src/utils/notifications.py
_notifications = []  # Global storage for testing

def notify_cli_failure(operation: str, error_message: str):
    """Notify about CLI failure and store notification."""
    notification = {
        "type": "cli_failure",
        "operation": operation,
        "message": error_message,
        "timestamp": time.time(),
        "severity": "error"
    }
    _notifications.append(notification)  # Store for testing
    logger.warning(f"❌ Claude Code CLI Failure: {operation}")

def get_notifications() -> List[Dict]:
    """Get all notifications for testing."""
    return _notifications.copy()

def clear_notifications():
    """Clear all notifications for testing."""
    _notifications.clear()
```

### Step 4: Fix Configuration Validator Logic
```python
# File: src/utils/config_validator.py
def validate_config_file(self) -> Dict[str, Any]:
    """Validate configuration file with correct logic."""
    try:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections
        required_sections = ['enterprise_coding_agent', 'components']
        errors = []
        
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        return {
            "valid": len(errors) == 0,  # Fix: Return True when no errors
            "errors": errors
        }
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}

def validate_security(self) -> Dict[str, Any]:
    """Validate security settings with proper checks."""
    insecure_settings = []
    
    # Check for auto_mode in config
    if self.config.get('auto_mode', False):
        insecure_settings.append("auto_mode enabled")
    
    # Check for other security issues
    if self.config.get('dangerous_permissions', False):
        insecure_settings.append("dangerous_permissions enabled")
    
    return {
        "secure": len(insecure_settings) == 0,
        "insecure_settings": insecure_settings
    }
```

### Step 5: Fix Test Mocking and Assertions
```python
# File: tests/test_enhanced_features.py
class TestUsageMonitor(unittest.TestCase):
    def setUp(self):
        self.usage_monitor = UsageMonitor()
        self.usage_monitor._reset_pause_state()  # Ensure clean state
        
    def test_usage_recording(self):
        """Test usage recording with clean state."""
        # Ensure monitor is not paused
        self.assertFalse(self.usage_monitor.paused)
        
        # Record several requests
        for i in range(5):
            success = self.usage_monitor.record_request("Coder", f"operation_{i}", 100)
            self.assertTrue(success)  # Should succeed now

class TestNotificationSystem(unittest.TestCase):
    def setUp(self):
        clear_notifications()  # Clear any existing notifications
        
    def test_cli_failure_notification(self):
        """Test CLI failure notification creation."""
        notify_cli_failure("test_operation", "Connection failed")
        notifications = get_notifications()
        self.assertEqual(len(notifications), 1)  # Should have 1 notification
        self.assertEqual(notifications[0]["type"], "cli_failure")
```

## TESTING STRATEGY

### Phase 1 Testing:
```bash
# Test usage monitor fixes
python -m pytest tests/test_enhanced_features.py::TestUsageMonitor -v

# Test authentication manager fixes  
python -m pytest tests/test_auth_manager.py::TestClaudeAuthManager -v
```

### Phase 2 Testing:
```bash
# Test notification system fixes
python -m pytest tests/test_enhanced_features.py::TestNotificationSystem -v

# Test configuration validator fixes
python -m pytest tests/test_enhanced_features.py::TestConfigValidator -v
```

### Phase 3 Testing:
```bash
# Test claude code provider fixes
python -m pytest tests/test_claude_code_provider.py -v

# Run full test suite
python -m pytest tests/ --tb=short -q
```

## EXPECTED OUTCOMES

After implementing these fixes:
- **Usage Monitor Tests**: 4 tests should pass (currently failing due to pause state)
- **Authentication Manager Tests**: 2 tests should pass (currently failing due to string mismatches)
- **Notification System Tests**: 2 tests should pass (currently failing due to missing notifications)
- **Configuration Validator Tests**: 2 tests should pass (currently failing due to validation logic)
- **Claude Code Provider Tests**: 4 tests should pass (currently failing due to usage limits)

**Total Expected Improvement**: 16 failing tests → 0 failing tests
**Final Test Status**: 103 passing, 0 failing, 2 skipped

## IMPLEMENTATION ORDER

1. **Start with Usage Monitor** (highest impact - fixes 5 tests)
2. **Fix Authentication Manager** (fixes 2 tests)
3. **Fix Notification System** (fixes 2 tests)
4. **Fix Configuration Validator** (fixes 2 tests)
5. **Fix Claude Code Provider** (fixes 4 tests)
6. **Run comprehensive test suite** to verify all fixes

This systematic approach ensures each fix builds on the previous ones and maximizes the number of tests fixed with minimal code changes.

## CLAUDE CODE PROMPT

Here's the exact prompt to give to Claude Code:

---

**TASK**: Fix all 16 failing tests in the Enterprise Agent codebase using the systematic approach outlined above.

**CONTEXT**: The Enterprise Agent has 16 failing tests out of 103 total tests. The failures are primarily due to:
1. Usage monitor being in paused state (5 tests)
2. String assertion mismatches in auth manager (2 tests)  
3. Missing notification storage (2 tests)
4. Configuration validation logic errors (2 tests)
5. Usage limit blocking in claude code provider (4 tests)

**APPROACH**: Follow the 5-phase systematic approach:
1. Fix Usage Monitor pause state first (highest impact)
2. Fix Authentication Manager string mismatches
3. Fix Notification System storage
4. Fix Configuration Validator logic
5. Fix Claude Code Provider mocking

**FILES TO MODIFY**:
- `src/utils/usage_monitor.py` - Add `_reset_pause_state()` method
- `src/providers/auth_manager.py` - Fix return strings in `verify_subscription_plan()`
- `src/utils/notifications.py` - Add notification storage functions
- `src/utils/config_validator.py` - Fix validation logic
- `tests/test_enhanced_features.py` - Add proper test setup
- `tests/test_claude_code_provider.py` - Fix mocking and assertions

**SUCCESS CRITERIA**: All 16 failing tests should pass, resulting in 103 passing tests, 0 failing tests, 2 skipped tests.

**IMPLEMENTATION**: Follow the detailed code examples provided in each phase. Test each phase before moving to the next. Use the testing strategy to verify fixes incrementally.

---
