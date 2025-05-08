# recommendations/management/commands/update_recommendations.py

from django.core.management.base import BaseCommand
from django.core.management import call_command

class Command(BaseCommand):
    help = 'Run the complete recommendation pipeline'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user_id',
            type=int,
            help='Process recommendations for a specific user (by ID)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration of all data'
        )

    def handle(self, *args, **options):
        user_id = options.get('user_id')
        force = options.get('force', False)
        
        user_args = ['--user_id', str(user_id)] if user_id else []
        force_args = ['--force'] if force else []
        
        self.stdout.write(self.style.NOTICE("Step 1: Generating post embeddings..."))
        call_command('generate_post_embeddings', *force_args)
        
        self.stdout.write(self.style.NOTICE("Step 2: Generating user Node2Vec vectors..."))
        call_command('generate_user_node2vec', *(user_args + force_args))
        
        self.stdout.write(self.style.NOTICE("Step 3: Generating recommendations..."))
        call_command('generate_recommendations', *(user_args + force_args))
        
        self.stdout.write(self.style.SUCCESS("Recommendation pipeline completed successfully!"))