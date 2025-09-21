#!/usr/bin/env python3
"""Test script to validate enhanced caching with configurable controls."""
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_cache_config():
    """Test cache configuration from environment and dict."""
    print("Testing cache configuration...")

    try:
        from src.utils.cache import CacheConfig

        # Test default configuration
        default_config = CacheConfig()
        assert default_config.enabled is True
        assert default_config.default_ttl == 300
        assert default_config.adaptive_ttl is False

        # Test environment configuration
        os.environ["CACHE_ENABLED"] = "false"
        os.environ["CACHE_DEFAULT_TTL"] = "600"
        os.environ["CACHE_ADAPTIVE_TTL"] = "true"

        env_config = CacheConfig.from_env()
        assert env_config.enabled is False
        assert env_config.default_ttl == 600.0
        assert env_config.adaptive_ttl is True

        # Test dict configuration
        dict_config = CacheConfig.from_dict(
            {
                "enabled": True,
                "default_ttl": 450,
                "max_size": 2000,
                "eviction_policy": "lfu",
            }
        )
        assert dict_config.enabled is True
        assert dict_config.default_ttl == 450
        assert dict_config.max_size == 2000
        assert dict_config.eviction_policy == "lfu"

        print("✅ Cache configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Cache configuration test failed: {e}")
        return False

    finally:
        # Cleanup environment variables
        os.environ.pop("CACHE_ENABLED", None)
        os.environ.pop("CACHE_DEFAULT_TTL", None)
        os.environ.pop("CACHE_ADAPTIVE_TTL", None)


def test_adaptive_ttl():
    """Test adaptive TTL based on quality scores."""
    print("\nTesting adaptive TTL...")

    try:
        from src.utils.cache import CacheConfig, TTLCache

        # Create cache with adaptive TTL enabled
        config = CacheConfig(
            adaptive_ttl=True,
            quality_threshold=0.8,
            high_quality_ttl_multiplier=2.0,
            low_quality_ttl_multiplier=0.5,
            default_ttl=100,
        )

        cache = TTLCache(config=config)

        # Test high quality item (should get extended TTL)
        cache.set("high_quality", "value1", quality_score=0.9)
        entry = cache._cache["high_quality"]
        assert entry.ttl == 200.0  # 100 * 2.0

        # Test low quality item (should get reduced TTL)
        cache.set("low_quality", "value2", quality_score=0.3)
        entry = cache._cache["low_quality"]
        assert entry.ttl == 50.0  # 100 * 0.5

        # Test medium quality item (should get normal TTL)
        cache.set("medium_quality", "value3", quality_score=0.6)
        entry = cache._cache["medium_quality"]
        assert entry.ttl == 50.0  # Still low quality (< 0.8)

        print("✅ Adaptive TTL test passed")
        return True

    except Exception as e:
        print(f"❌ Adaptive TTL test failed: {e}")
        return False


def test_eviction_policies():
    """Test different eviction policies."""
    print("\nTesting eviction policies...")

    try:
        from src.utils.cache import CacheConfig, TTLCache

        # Test LRU eviction
        lru_config = CacheConfig(max_size=3, eviction_policy="lru")
        lru_cache = TTLCache(config=lru_config)

        lru_cache.set("a", "value_a")
        lru_cache.set("b", "value_b")
        lru_cache.set("c", "value_c")

        # Access 'a' to make it most recently used
        lru_cache.get("a")

        # Add new item, should evict 'b' (least recently used)
        lru_cache.set("d", "value_d")

        assert lru_cache.get("a") == "value_a"  # Should still exist
        assert lru_cache.get("b") is None  # Should be evicted
        assert lru_cache.get("c") == "value_c"  # Should still exist
        assert lru_cache.get("d") == "value_d"  # Should exist

        # Test LFU eviction
        lfu_config = CacheConfig(max_size=3, eviction_policy="lfu")
        lfu_cache = TTLCache(config=lfu_config)

        lfu_cache.set("x", "value_x")
        lfu_cache.set("y", "value_y")
        lfu_cache.set("z", "value_z")

        # Access 'x' multiple times to increase frequency
        lfu_cache.get("x")
        lfu_cache.get("x")
        lfu_cache.get("y")  # Access 'y' once

        # Add new item, should evict 'z' (least frequently used)
        lfu_cache.set("w", "value_w")

        assert lfu_cache.get("x") == "value_x"  # Should still exist (most frequent)
        assert lfu_cache.get("y") == "value_y"  # Should still exist
        assert lfu_cache.get("z") is None  # Should be evicted (least frequent)
        assert lfu_cache.get("w") == "value_w"  # Should exist

        print("✅ Eviction policies test passed")
        return True

    except Exception as e:
        print(f"❌ Eviction policies test failed: {e}")
        return False


def test_compression():
    """Test compression functionality."""
    print("\nTesting compression...")

    try:
        from src.utils.cache import CacheConfig, TTLCache

        # Create cache with compression enabled
        config = CacheConfig(
            compression_enabled=True,
            compression_threshold=50,  # Small threshold for testing
        )

        cache = TTLCache(config=config)

        # Small value (should not be compressed)
        small_value = "small"
        cache.set("small", small_value)
        assert cache.get("small") == small_value

        # Large value (should be compressed)
        large_value = "x" * 100  # Larger than threshold
        cache.set("large", large_value)
        retrieved_value = cache.get("large")
        assert retrieved_value == large_value

        # Check compression stats
        stats = cache.get_stats()
        assert stats.get("compression_saves", 0) > 0

        print("✅ Compression test passed")
        return True

    except Exception as e:
        print(f"❌ Compression test failed: {e}")
        return False


def test_persistence():
    """Test persistence functionality."""
    print("\nTesting persistence...")

    try:
        from src.utils.cache import CacheConfig, TTLCache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with persistence enabled
            config = CacheConfig(
                persistence_enabled=True,
                persistence_path=temp_dir,
                default_ttl=3600,  # Long TTL to avoid expiration during test
            )

            # Create first cache instance and add data
            cache1 = TTLCache(config=config)
            cache1.set("persistent_key", "persistent_value")
            assert cache1.get("persistent_key") == "persistent_value"

            # Create second cache instance (should load from disk)
            cache2 = TTLCache(config=config)
            assert cache2.get("persistent_key") == "persistent_value"

        print("✅ Persistence test passed")
        return True

    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False


def test_cache_disabled():
    """Test cache behavior when disabled."""
    print("\nTesting cache disabled behavior...")

    try:
        from src.utils.cache import CacheConfig, TTLCache

        # Create cache with caching disabled
        config = CacheConfig(enabled=False)
        cache = TTLCache(config=config)

        # Set value (should not be stored)
        cache.set("key", "value")

        # Get value (should return default)
        assert cache.get("key") is None
        assert cache.get("key", "default") == "default"

        print("✅ Cache disabled test passed")
        return True

    except Exception as e:
        print(f"❌ Cache disabled test failed: {e}")
        return False


def test_enhanced_model_cache():
    """Test enhanced model response cache."""
    print("\nTesting enhanced model cache...")

    try:
        from src.utils.cache import CacheConfig, ModelResponseCache

        config = CacheConfig(
            adaptive_ttl=True,
            quality_threshold=0.8,
            high_quality_ttl_multiplier=2.0,
            default_ttl=100,
        )

        model_cache = ModelResponseCache(config=config)

        # Cache high-quality response
        model_cache.cache_response(
            model="claude-3",
            prompt="Test prompt",
            response="High quality response",
            ttl=100,
        )

        # Update quality score (should extend TTL if adaptive)
        key = model_cache._create_key("claude-3", "Test prompt")
        model_cache.update_quality(key, 0.9)  # High quality

        # Verify response is cached
        cached_response = model_cache.get_response("claude-3", "Test prompt")
        assert cached_response == "High quality response"

        print("✅ Enhanced model cache test passed")
        return True

    except Exception as e:
        print(f"❌ Enhanced model cache test failed: {e}")
        return False


def main():
    """Run all enhanced caching tests."""
    print("Enhanced Caching with Configurable Controls Test Suite")
    print("=" * 60)

    tests_passed = 0
    total_tests = 7

    if test_cache_config():
        tests_passed += 1

    if test_adaptive_ttl():
        tests_passed += 1

    if test_eviction_policies():
        tests_passed += 1

    if test_compression():
        tests_passed += 1

    if test_persistence():
        tests_passed += 1

    if test_cache_disabled():
        tests_passed += 1

    if test_enhanced_model_cache():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print(
            "🎉 All enhanced caching tests passed! Configurable controls are working correctly."
        )
        return 0
    else:
        print("❌ Some tests failed. Please check the caching implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
