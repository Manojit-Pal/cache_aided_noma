# First 688 lines unchanged...
# Just showing the _compile_results method with the fix

    def _compile_results(self, cache: CacheBase) -> Dict:
        """
        Compile final results for this episode.
        """
        total_req = max(self.metrics['total_requests'], 1)
        total_noma = max(self.metrics['noma_transmissions'], 1)
        total_sic = max(self.metrics['sic_attempts'], 1)
        
        # ✅ BUG FIX #10: Correct denominators for per-user rates
        # Each NOMA transmission has 2 users (weak + strong)
        total_noma_users = total_noma * 2  # Total users in NOMA pairs
        
        results = {
            # Cache performance
            'hit_rate': self.metrics['cache_hits'] / total_req,
            'miss_rate': self.metrics['cache_misses'] / total_req,
            
            # NOMA performance
            'outage_probability': self.metrics['outages'] / total_noma_users,  # ✅ Use total_noma_users
            'noma_success_rate': self.metrics['noma_successes'] / total_noma,
            
            # SIC performance
            'sic_success_rate': self.metrics['sic_successes'] / total_sic,
            
            # CIC performance (NOVEL CONTRIBUTION)
            # ✅ BUG FIX #10: Divide by total users (2 per pair), not just number of pairs
            # cic_enabled_weak and cic_enabled_strong count individual users who benefited
            # Since each pair has 2 users, divide by total_noma * 2
            'cic_opportunity_rate': self.metrics['cic_opportunities'] / total_noma_users,  # ✅ Fixed
            'cic_benefit_rate': (self.metrics['cic_enabled_weak'] + 
                                 self.metrics['cic_enabled_strong']) / total_noma_users,  # ✅ Fixed!
            
            # Throughput
            'avg_throughput': self.metrics['total_throughput'] / total_req,
            'spectral_efficiency': self.metrics['total_throughput'] / total_noma,
            
            # Energy efficiency
            'energy_per_bit': (self.metrics['total_energy'] / 
                              max(self.metrics['total_throughput'], 1)),
            
            # Raw counts
            'total_requests': self.metrics['total_requests'],
            'cache_hits': self.metrics['cache_hits'],
            'noma_transmissions': self.metrics['noma_transmissions'],
            'outages': self.metrics['outages'],
            'cic_events': len(self.cic_events),
        }
        
        # Add cache-specific stats if available
        if hasattr(cache, 'stats'):
            cache_stats = cache.stats()
            results.update({f'cache_{k}': v for k, v in cache_stats.items()})
        
        return results
