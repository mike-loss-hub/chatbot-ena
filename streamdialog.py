import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

st.set_page_config(page_title="S3 File Selector", layout="wide")

# Apply dark theme styling
st.markdown("""
<style>
    /* Dark theme styles */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #262730;
    }
    /* Button styling */
    .stButton>button {
        background-color: transparent;
        border: none;
        color: #4F8BF9;
        margin-top: 0px;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #262730;
        color: #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

st.title("S3 File Selector")

# Create a text input for the S3 URI
s3_uri = st.text_input("S3 URI", key="s3_uri")

# Create a button to open the S3 browser
if st.button("Browse S3"):
    # Create a container for the S3 browser dialog
    dialog = st.container()
    
    with dialog:
        st.subheader("Select a file from S3")
        
        # Initialize S3 client
        try:
            s3 = boto3.client('s3')
            
            # Get list of buckets
            try:
                response = s3.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]
                
                if not buckets:
                    st.warning("No S3 buckets found in your account.")
                else:
                    # Select a bucket
                    selected_bucket = st.selectbox("Select S3 Bucket", buckets)
                    
                    if selected_bucket:
                        # Optional prefix filter
                        prefix = st.text_input("Filter by prefix (optional)")
                        
                        # List objects in the bucket
                        try:
                            paginator = s3.get_paginator('list_objects_v2')
                            
                            # Display a progress message while loading
                            with st.spinner(f"Loading files from {selected_bucket}..."):
                                # Get all objects with pagination
                                files = []
                                for page in paginator.paginate(Bucket=selected_bucket, Prefix=prefix):
                                    if 'Contents' in page:
                                        for obj in page['Contents']:
                                            files.append(obj['Key'])
                            
                            if not files:
                                st.info(f"No files found in bucket '{selected_bucket}' with prefix '{prefix}'")
                            else:
                                # Show number of files found
                                st.success(f"Found {len(files)} files")
                                
                                # Add search filter
                                search = st.text_input("Search files")
                                if search:
                                    files = [f for f in files if search.lower() in f.lower()]
                                    st.write(f"Found {len(files)} matching files")
                                
                                # Select a file
                                selected_file = st.selectbox("Select a file", files)
                                
                                if selected_file:
                                    # Construct the S3 URI
                                    uri = f"s3://{selected_bucket}/{selected_file}"
                                    
                                    # Display file info
                                    try:
                                        response = s3.head_object(Bucket=selected_bucket, Key=selected_file)
                                        size_mb = response['ContentLength'] / (1024 * 1024)
                                        last_modified = response['LastModified']
                                        
                                        st.write(f"Size: {size_mb:.2f} MB")
                                        st.write(f"Last modified: {last_modified}")
                                    except Exception as e:
                                        st.error(f"Error getting file info: {str(e)}")
                                    
                                    # Button to select this file
                                    if st.button("Select this file"):
                                        # Update the text input with the S3 URI
                                        st.session_state.s3_uri = uri
                                        st.experimental_rerun()
                        
                        except ClientError as e:
                            st.error(f"Error listing objects: {str(e)}")
            
            except ClientError as e:
                st.error(f"Error listing buckets: {str(e)}")
        
        except NoCredentialsError:
            st.error("AWS credentials not found. Please configure your AWS credentials.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Display the selected S3 URI
if s3_uri:
    st.success("Selected file:")
    st.code(s3_uri)
    
    # Add a button to clear the selection
    if st.button("Clear selection"):
        st.session_state.s3_uri = ""
        st.experimental_rerun()

# Add some helpful information at the bottom
st.markdown("---")
st.markdown("""
**Note:** This app requires AWS credentials with S3 read permissions.
Make sure you have configured your AWS credentials properly.
""")
