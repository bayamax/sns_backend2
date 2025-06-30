# accounts/serializers.py

from rest_framework import serializers
from django.contrib.auth import get_user_model, authenticate
from django.utils.translation import gettext_lazy as _
from .models import Follow, Block

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    """ユーザーシリアライザー"""
    followers_count = serializers.SerializerMethodField()
    following_count = serializers.SerializerMethodField()
    profile_image_url = serializers.SerializerMethodField()
    is_blocked_by_me = serializers.SerializerMethodField()
    am_i_blocked = serializers.SerializerMethodField()
    is_following = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'profile_image_url', 'bio', 
                  'followers_count', 'following_count', 'is_blocked_by_me', 'am_i_blocked', 'is_following']
        read_only_fields = ['id', 'followers_count', 'following_count']
    
    def get_followers_count(self, obj):
        return obj.followers_count
    
    def get_following_count(self, obj):
        return obj.following_count
    
    def get_profile_image_url(self, obj):
        if obj.profile_image and hasattr(obj.profile_image, 'url'):
            request = self.context.get('request')
            if request:
                # 修正点: build_absolute_url ではなく build_absolute_uri を使用
                return request.build_absolute_uri(obj.profile_image_url)
        return None

    def get_is_blocked_by_me(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            # blocker=リクエストユーザー, blocked=プロフィール対象ユーザー で検索
            return Block.objects.filter(blocker=request.user, blocked=obj).exists()
        return False

    def get_am_i_blocked(self, obj):
        request = self.context.get('request')
        print(f"--- Debug: get_am_i_blocked ---")
        print(f"    Profile User (obj.id): {obj.id}")
        if request and request.user.is_authenticated:
            print(f"    Request User (request.user.id): {request.user.id}")
            # blocker=プロフィール対象ユーザー, blocked=リクエストユーザー で検索
            exists = Block.objects.filter(blocker=obj, blocked=request.user).exists()
            print(f"    Block record exists (blocker={obj.id}, blocked={request.user.id}): {exists}")
            return exists
        else:
            print(f"    Request user not authenticated or request context missing.")
            return False

    def get_is_following(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return Follow.objects.filter(follower=request.user, following=obj).exists()
        return False

class UserRegistrationSerializer(serializers.ModelSerializer):
    """ユーザー登録シリアライザー"""
    password = serializers.CharField(style={'input_type': 'password'}, write_only=True)
    password2 = serializers.CharField(style={'input_type': 'password'}, write_only=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2']
    
    def validate(self, attrs):
        # パスワード確認
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError(_("Passwords don't match."))
        
        # ユーザー名の重複チェック
        username = attrs.get('username')
        if User.objects.filter(username__iexact=username).exists():
            raise serializers.ValidationError(_("Username already exists."))
        
        return attrs
    
    def validate_password(self, value):
        # パスワードポリシーのチェック
        if len(value) < 8:
            raise serializers.ValidationError(_("Password must be at least 8 characters."))
        if len(value) > 64:
            raise serializers.ValidationError(_("Password must be less than 64 characters."))
        if not any(char.isdigit() for char in value):
            raise serializers.ValidationError(_("Password must contain at least one number."))
        if not any(char.isupper() for char in value):
            raise serializers.ValidationError(_("Password must contain at least one uppercase letter."))
        if not any(char.islower() for char in value):
            raise serializers.ValidationError(_("Password must contain at least one lowercase letter."))
        
        # 一般的なパスワードのチェック（実際はもっと多くのパスワードリストを持つべき）
        common_passwords = ['password', 'qwerty', '123456', 'admin', 'welcome']
        if value.lower() in common_passwords:
            raise serializers.ValidationError(_("This password is too common."))
        
        return value
    
    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password']
        )
        return user

class UserLoginSerializer(serializers.Serializer):
    """ユーザーログインシリアライザー"""
    username = serializers.CharField()
    password = serializers.CharField(style={'input_type': 'password'}, write_only=True)
    
    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')
        
        if username and password:
            user = authenticate(username=username, password=password)
            
            if not user:
                # メールアドレスでの認証も試す
                try:
                    user_obj = User.objects.get(email=username)
                    user = authenticate(username=user_obj.username, password=password)
                except User.DoesNotExist:
                    user = None
            
            if not user:
                msg = _('Unable to log in with provided credentials.')
                raise serializers.ValidationError(msg, code='authorization')
        else:
            msg = _('Must include "username/email" and "password".')
            raise serializers.ValidationError(msg, code='authorization')
        
        attrs['user'] = user
        return attrs

class UserProfileUpdateSerializer(serializers.ModelSerializer):
    """ユーザープロフィール更新シリアライザー"""
    class Meta:
        model = User
        fields = ['username', 'bio', 'profile_image']
    
    def validate_username(self, value):
        user = self.context['request'].user
        if User.objects.exclude(pk=user.pk).filter(username__iexact=value).exists():
            raise serializers.ValidationError(_("This username is already in use."))
        return value
    
    def update(self, instance, validated_data):
        instance.username = validated_data.get('username', instance.username)
        instance.bio = validated_data.get('bio', instance.bio)
        
        if 'profile_image' in validated_data:
            # 古い画像が存在する場合は削除
            if instance.profile_image:
                instance.profile_image.delete(save=False)
            instance.profile_image = validated_data.get('profile_image')
        
        instance.save()
        return instance

class FollowSerializer(serializers.ModelSerializer):
    """フォロー関係シリアライザー"""
    follower = serializers.PrimaryKeyRelatedField(read_only=True)
    following = serializers.PrimaryKeyRelatedField(queryset=User.objects.all())
    
    class Meta:
        model = Follow
        fields = ['follower', 'following', 'created_at']
        read_only_fields = ['created_at']
    
    def create(self, validated_data):
        follower = self.context['request'].user
        following = validated_data['following']
        
        # 自分自身をフォローしようとした場合
        if follower == following:
            raise serializers.ValidationError(_("You cannot follow yourself."))
        
        # すでにフォローしている場合
        if Follow.objects.filter(follower=follower, following=following).exists():
            raise serializers.ValidationError(_("You are already following this user."))
        
        follow = Follow.objects.create(follower=follower, following=following)
        return follow

class UserFollowersSerializer(serializers.ModelSerializer):
    """ユーザーのフォロワーリストシリアライザー"""
    user = UserSerializer(source='follower')
    
    class Meta:
        model = Follow
        fields = ['user', 'created_at']

class UserFollowingSerializer(serializers.ModelSerializer):
    """ユーザーのフォローリストシリアライザー"""
    user = UserSerializer(source='following')
    
    class Meta:
        model = Follow
        fields = ['user', 'created_at']

# accounts/serializers.py の末尾に追加
RegisterSerializer = UserRegistrationSerializer
LoginSerializer = UserLoginSerializer
UserProfileSerializer = UserProfileUpdateSerializer

class UserListSerializer(serializers.ModelSerializer):
    """ユーザーリスト表示用シリアライザー"""
    profile_image_url = serializers.SerializerMethodField()
    followers_count = serializers.SerializerMethodField()
    following_count = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'profile_image_url', 'bio', 'followers_count', 'following_count']
    
    def get_profile_image_url(self, obj):
        if obj.profile_image and hasattr(obj.profile_image, 'url'):
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.profile_image.url)
            return obj.profile_image.url
        return None
    
    def get_followers_count(self, obj):
        return Follow.objects.filter(following=obj).count()
    
    def get_following_count(self, obj):
        return Follow.objects.filter(follower=obj).count()


class AppleLoginSerializer(serializers.Serializer):
    """Apple ID ログイン用シリアライザー"""
    identity_token = serializers.CharField()
    username = serializers.CharField(required=False, allow_blank=True)

